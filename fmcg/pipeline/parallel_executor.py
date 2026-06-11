"""
Multi-GPU parallel execution infrastructure for fMCG pipeline.

This module provides utilities for running pipeline jobs across multiple GPUs with
proper resource cleanup, signal handling, and progress tracking.
"""

import os
import signal
import atexit
import multiprocessing
import psutil
import torch
import gc
from typing import List, Optional, Callable, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class MultiGPUExecutor:
    """Manager for multi-GPU parallel execution with cleanup and signal handling.
    
    This class handles:
    - Round-robin GPU device assignment
    - Process pool management
    - Signal handling (Ctrl+C)
    - GPU memory cleanup
    - Progress tracking
    
    Example:
        >>> executor = MultiGPUExecutor(devices=["cuda:0", "cuda:1"], workers_per_device=2)
        >>> results = executor.run_jobs(jobs, worker_fn)
    """
    
    def __init__(self, devices: List[str], workers_per_device: int = 1, use_spawn: bool = True):
        """Initialize multi-GPU executor.
        
        Args:
            devices: List of GPU device strings (e.g., ["cuda:0", "cuda:1"])
            workers_per_device: Number of parallel workers per GPU
            use_spawn: Use 'spawn' start method for child processes (default True).
                Required when CUDA has been initialized in the main process
                before forking (e.g. a reference Pipeline run).
        """
        self.devices = devices
        self.workers_per_device = workers_per_device
        self.max_workers = len(devices) * workers_per_device
        self._mp_context = multiprocessing.get_context('spawn') if use_spawn else None
        
        # Global state for cleanup
        self._active_executor = None
        self._parent_pid = os.getpid()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n\n{'='*80}")
            print(f"⚠️  INTERRUPT SIGNAL RECEIVED - CLEANING UP")
            print(f"{'='*80}")
            
            # Force kill all processes
            self._cleanup_resources(force_kill=True)
            
            print(f"{'='*80}")
            print("Exiting...")
            print(f"{'='*80}\n")
            
            # Exit immediately
            os._exit(1)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Register cleanup on normal exit
        def normal_exit_cleanup():
            if self._active_executor is not None:
                self._cleanup_resources(force_kill=False)
        
        atexit.register(normal_exit_cleanup)
    
    def _cleanup_resources(self, force_kill: bool = False):
        """Clean up GPU resources and active processes.
        
        Args:
            force_kill: If True, forcefully kill all child processes
        """
        # Kill all child processes forcefully
        if force_kill:
            print("\nForce killing all worker processes...")
            try:
                parent = psutil.Process(self._parent_pid)
                children = parent.children(recursive=True)
                
                if children:
                    print(f"Found {len(children)} child processes to terminate")
                    
                    # First try SIGTERM
                    for child in children:
                        try:
                            print(f"  Terminating PID {child.pid}...")
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    # Wait up to 3 seconds
                    gone, alive = psutil.wait_procs(children, timeout=3)
                    
                    # Force kill remaining processes
                    if alive:
                        print(f"Force killing {len(alive)} remaining processes...")
                        for child in alive:
                            try:
                                print(f"  Killing PID {child.pid}...")
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # Wait for kills to complete
                        psutil.wait_procs(alive, timeout=2)
                    
                    print(f"✓ All worker processes terminated")
                else:
                    print("No child processes found")
                    
            except Exception as e:
                print(f"Error during process cleanup: {e}")
        
        # Shutdown executor if active
        if self._active_executor is not None:
            try:
                print("Shutting down executor...")
                self._active_executor.shutdown(wait=False, cancel_futures=True)
                self._active_executor = None
            except Exception as e:
                print(f"Error shutting down executor: {e}")
        
        # Clean up GPU devices
        cleanup_gpu_memory(self.devices)
    
    def run_jobs(
        self,
        jobs: List[Dict[str, Any]],
        worker_fn: Callable,
        progress_desc: str = "Processing",
        progress_ncols: int = 100
    ) -> List[Dict[str, Any]]:
        """Run jobs in parallel across GPUs.
        
        Args:
            jobs: List of job dictionaries to process
            worker_fn: Worker function that processes a single job
            progress_desc: Description for progress bar
            progress_ncols: Width of progress bar
            
        Returns:
            List of result dictionaries from worker_fn
        """
        if not jobs:
            print("No jobs to process.")
            return []
        
        print(f"\n{'='*80}")
        print(f"Starting with {len(jobs)} jobs across {self.max_workers} workers")
        print(f"Devices: {self.devices} ({self.workers_per_device} workers per device)")
        print(f"Press Ctrl+C to cancel and clean up safely")
        print(f"{'='*80}\n")
        
        results = []
        success_count = 0
        failure_count = 0
        
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=self._mp_context,
            ) as executor:
                # Register executor for signal handling
                self._active_executor = executor
                
                # Submit all jobs
                future_to_job = {executor.submit(worker_fn, job): job for job in jobs}
                
                # Collect results with progress bar
                with tqdm(total=len(jobs), desc=progress_desc, ncols=progress_ncols,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result.get("status") == "success":
                                success_count += 1
                            else:
                                failure_count += 1
                            
                            # Update progress bar with custom postfix
                            postfix = self._format_progress_postfix(result, success_count, failure_count)
                            pbar.set_postfix_str(postfix)
                            
                        except Exception as e:
                            failure_count += 1
                            print(f"✗ Failed to get result: {e}")
                            results.append({
                                "status": "failed",
                                "error": str(e),
                                **{k: v for k, v in job.items() if k in ['patient', 'series', 'record', 'device']}
                            })
                        
                        pbar.update(1)
                
                # Clear executor reference after successful completion
                self._active_executor = None
        
        except KeyboardInterrupt:
            print("\n\n⚠️  KeyboardInterrupt - force cleaning up...")
            self._cleanup_resources(force_kill=True)
            raise
        except Exception as e:
            print(f"\n\n⚠️  Exception during execution: {e}")
            self._cleanup_resources(force_kill=True)
            raise
        
        print(f"\n{'='*80}")
        print(f"Execution complete: {success_count} successful, {failure_count} failed")
        print(f"{'='*80}\n")
        
        # Final cleanup after all jobs complete
        self._cleanup_resources(force_kill=False)
        
        return results
    
    def _format_progress_postfix(self, result: Dict[str, Any], success_count: int, failure_count: int) -> str:
        """Format progress bar postfix string.
        
        Args:
            result: Result dictionary from worker
            success_count: Number of successful jobs
            failure_count: Number of failed jobs
            
        Returns:
            Formatted postfix string
        """
        device = result.get('device', 'N/A')
        
        # Try to create a short identifier
        identifiers = []
        for key in ['patient', 'series', 'record', 'config_name']:
            if key in result:
                identifiers.append(str(result[key]))
        
        if identifiers:
            job_id = '_'.join(identifiers)
            if len(job_id) > 40:
                job_id = job_id[:37] + "..."
        else:
            job_id = "job"
        
        return f"✓{success_count} ✗{failure_count} | {job_id} [{device}]"


def cleanup_gpu_memory(devices: Optional[List[str]] = None):
    """Clean up GPU memory for specified devices.
    
    Args:
        devices: List of device strings (e.g., ["cuda:0", "cuda:1"]).
                If None, skips GPU cleanup.
    """
    if torch.cuda.is_available() and devices:
        print("Cleaning up GPU memory...")
        for device in devices:
            try:
                device_id = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(device_id)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print(f"  ✓ Cleaned {device}")
            except Exception as e:
                print(f"  ✗ Failed to clean up {device}: {e}")
    
    # Force garbage collection
    gc.collect()
    gc.collect()
    if devices:
        print("✓ Cleanup complete\n")


def cleanup_worker_process(device: str):
    """Clean up GPU memory in a worker process.
    
    Call this at the end of worker functions to ensure proper cleanup.
    
    Args:
        device: Device string (e.g., "cuda:0")
    """
    if torch.cuda.is_available() and device.startswith('cuda'):
        try:
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass  # Ignore cleanup errors
    
    # Force garbage collection
    gc.collect()
    gc.collect()


def suppress_worker_output():
    """Suppress stdout/stderr in worker processes to avoid mixed output.
    
    Returns:
        Tuple of (old_stdout, old_stderr) for restoration
        
    Example:
        >>> old_stdout, old_stderr = suppress_worker_output()
        >>> # ... do work ...
        >>> sys.stdout, sys.stderr = old_stdout, old_stderr
    """
    import sys
    import io
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    return old_stdout, old_stderr


def restore_worker_output(old_stdout, old_stderr):
    """Restore stdout/stderr in worker processes.
    
    Args:
        old_stdout: Original stdout
        old_stderr: Original stderr
    """
    import sys
    sys.stdout = old_stdout
    sys.stderr = old_stderr
