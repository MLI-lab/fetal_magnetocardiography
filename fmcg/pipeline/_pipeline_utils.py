import logging
import os
import pickle
import traceback

import numpy as np
import pandas as pd
import torch

from ..utils import data
from ..fitting.inverse_solver import InverseSolver
from ..analysis import heartbeats
from ..signal.filtering import wavelet_denoise

logger = logging.getLogger(__name__)

# --- Multiprocessing worker for overlapping window processing ---
def _process_window_worker(args):
    """
    Worker function for multiprocess window processing.

    Parameters
    ----------
    args : tuple
        (window_data, window_time, start, end, segment_idx, config, axis_mask, W, fs_, initial_parameters_slice, field_true_slice)
    """
    (
        window_data,
        window_time,
        start,
        end,
        segment_idx,
        config,
        axis_mask,
        W,
        fs_,
        initial_parameters_slice,
        field_true_slice,
    ) = args

    # Import here to avoid issues with multiprocessing
    from fmcg.fitting.inverse_solver import InverseSolver
    import torch
    import numpy as np

    N = window_data.shape[0]
    device = config.get("device", "cpu")
    
    # If field_true_slice is provided, use it (it's already scaled/processed)
    # Otherwise fall back to window_data (but this shouldn't happen with new logic)
    if field_true_slice is not None:
        y = torch.tensor(field_true_slice, dtype=torch.float32, device=device)
    else:
        y = torch.tensor(window_data, dtype=torch.float32, device=device)

    # Create solver for this window
    mdl = InverseSolver(
        data.generate_array_coordinates(grid_shape=(4, 4), grid_spacing=0.04, y=0),
        num_dipoles=config["solver"]["num_dipoles"],
        scaling=dict(m=config["solver"]["m_scaling"]),
        basis=None,  # If needed, pass basis config
        axis_mask=axis_mask,
        device="cpu",  # Force CPU for multiprocessing
        method=config["solver"]["optimizer"],
        whitening_matrix=W,
        verbose=False,
    )

    # Use provided initial parameters
    # Ensure they are on the correct device
    initial_parameters = {}
    if initial_parameters_slice is not None:
        for k, v in initial_parameters_slice.items():
            if v is not None:
                initial_parameters[k] = torch.tensor(v, dtype=torch.float32, device=device)
            else:
                initial_parameters[k] = None
    else:
        # Fallback to local initialization (should not be reached if logic is correct)
        r_init = np.repeat(config["solver"]["r_init"], N, axis=0)
        from ._pipeline_utils import _initialize_solver_parameters
        initial_parameters, y = _initialize_solver_parameters(mdl, r_init, y, config)

    from ._pipeline_utils import _solve_segment, _apply_wavelet_denoising
    
    # Solve
    m_hat, r_hat = _solve_segment(mdl, y, initial_parameters, config, segment_idx=segment_idx)
    m_hat = _apply_wavelet_denoising(m_hat, config)

    # Convert to numpy if needed
    if hasattr(m_hat, "cpu"):
        m_hat = m_hat.cpu().numpy()
    if hasattr(r_hat, "cpu"):
        r_hat = r_hat.cpu().numpy()

    return {
        'start': start,
        'end': end,
        'm_hat': m_hat,
        'r_hat': r_hat,
        'segment_idx': segment_idx
    }

def _create_basis_dict(config, N):
    """Create basis dictionary for solver configuration."""
    basis_dict = {}
    # Ensure r_basis is a dict and add n_time
    if "r_basis" in config["solver"]:
        r_basis = config["solver"]["r_basis"]
        if isinstance(r_basis, dict):
            basis_dict["r"] = dict(r_basis)  # shallow copy
            basis_dict["r"]["n_time"] = N
        else:
            basis_dict["r"] = {"basis": r_basis, "n_time": N}
        if "r_bnds" in config["solver"]:
            basis_dict["r"]["bnds"] = config["solver"]["r_bnds"]
    # Ensure m_basis is a dict and add n_time
    if "m_basis" in config["solver"]:
        m_basis = config["solver"]["m_basis"]
        if isinstance(m_basis, dict):
            basis_dict["m"] = dict(m_basis)  # shallow copy
            basis_dict["m"]["n_time"] = N
        else:
            basis_dict["m"] = {"basis": m_basis, "n_time": N}
    return basis_dict

def _create_inverse_solver(config, N, axis_mask, W):
    """Create and initialize InverseSolver instance."""

    # Determine device once from config
    device = config.get("device", "cpu")

    r_sensors = data.generate_array_coordinates(
        grid_shape=(4, 4), grid_spacing=0.04, y=0
    )
    num_dipoles = config["solver"]["num_dipoles"]
    basis_dict = _create_basis_dict(config, N)
    return InverseSolver(
        r_sensors,
        num_dipoles=num_dipoles,
        scaling=dict(m=config["solver"]["m_scaling"]),
        basis=basis_dict,
        axis_mask=axis_mask,
        device=device,
        method=config["solver"]["optimizer"],
        whitening_matrix=W,
        verbose=False,
    )


def _initialize_solver_parameters(mdl, r_init, y, config):
    """Initialize solver parameters."""
    device = config.get("device", "cpu")

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32, device=device)
    if not isinstance(r_init, torch.Tensor):
        r_init = torch.tensor(r_init, dtype=torch.float32, device=device)

    return mdl.initialize_parameters(
        r_init,
        y,
        field_scaling=config["solver"]["initializer"]["field_scaling"],
        update_m_scaling=config["solver"]["initializer"]["update_m_scaling"],
        method=config["solver"]["initializer"]["method"],
        m = torch.zeros(y.shape[0], mdl.num_dipoles, 3, device=device)*1e-12 if config["solver"]["initializer"]["method"] == "given" else None,
        rcond=config["solver"]["initializer"]["rcond"],
        determine_scaling=config["solver"]["initializer"]["determine_scaling"],
    )


def _solve_segment(mdl, field_true, initial_parameters, config, segment_idx=None, plot=False):
    """Solve the inverse problem for a segment."""
    return mdl.solve(
        field_true,
        initial_parameters,
        lr=config["solver"]["lr"],
        loss_params=config["solver"].get("loss_params", None),
        plot=plot,
        plt_param=False,
        savename=os.path.join(config["output_dir"], f"loss_{segment_idx}.pdf") if plot and segment_idx is not None else None
    )


def _apply_wavelet_denoising(m_hat, config):
    """Apply wavelet denoising if enabled."""
    if config["post_processing"]["wavelet_denoise"]:
        for i in range(m_hat.shape[1]):
            m_hat[:, i] = wavelet_denoise(
                m_hat[:, i],
                wavelet="db6",
                level=None,
                threshold=config["post_processing"]["wavelet_denoise_threshold"],
                verbose=False,
            )
    return m_hat


def _save_pipeline_data(output_dir, heartbeats_dict, data_dict):
    """Save heartbeats and data dictionaries."""
    with open(os.path.join(output_dir, "heartbeats_dict.pkl"), "wb") as f:
        pickle.dump(heartbeats_dict, f)
    with open(os.path.join(output_dir, "data_dict.pkl"), "wb") as f:
        pickle.dump(data_dict, f)


def _process_dipole_data(
    m_hat, r_hat, fs, config, log_dict, show_plots, artifacts_mask=None
):
    """Process dipole moments using post-processing pipeline."""
    
    # For enhanced segmented processing, we support two types of segmentation:
    # 1. Pipeline-level segmentation (process_segments): Based on artifact detection
    # 2. Post-processing segmentation (segment_length): Based on fixed time windows for ICA/LMS
    
    data_dict = heartbeats.detect_heartbeats(
        m_hat,
        r_hat,
        fs,
        show_plots=show_plots,
        method=config["post_processing"]["method"],
        decomposition=config["post_processing"].get("decomposition", "ica"),
        outlier_kwargs=config["post_processing"]["averaging"].get(
            "outlier_kwargs", {"win": 4, "threshold": 10}
        ),
        savename=None,
        log_dict=log_dict,
        n_trials=config["post_processing"]["n_trials"],
        segment_length=config["post_processing"].get("segment_length", None),
        ica_components=config["post_processing"]["ica_components"],
        mu=config["post_processing"]["mu"],
        tol=config["post_processing"]["averaging"]["tol"],
        nlms=config["post_processing"].get("nlms", False),
        verbose=False,
    )
    
    if artifacts_mask is not None and "fetal" in data_dict and "maternal" in data_dict:
        data_dict["fetal"]["artifacts_mask"] = artifacts_mask
        data_dict["maternal"]["artifacts_mask"] = artifacts_mask
    
    return data_dict


def _segment_and_average_beats(data_dict, fs, config, log_dict):
    """Segment and average heartbeats from processed dipole data."""
    
    # The segment_and_average function now automatically detects if data was processed
    # in segments (presence of 'segment_results') and routes to appropriate processing
    return heartbeats.segment_and_average_heartbeats(
        data_dict,
        fs,
        method=config["post_processing"]["averaging"]["method"],
        tol=config["post_processing"]["averaging"]["tol"],
        outlier_kwargs=config["post_processing"]["averaging"].get(
            "outlier_kwargs", {"win": 4, "threshold": 10}
        ),
        min_block_length=config["post_processing"].get("min_block_length", 5),
        ratio_pre=config["post_processing"].get("ratio_pre", 0.5),
        interval=config["post_processing"].get("interval", 2.5),
        log_dict=log_dict,
        verbose=False,
    )



def _plot_averaged_beats_if_enabled(heartbeats_dict, save_plots, output_dir):
    """Plot averaged beats if plotting is enabled."""
    if save_plots and heartbeats_dict:
        # Check if there are any beats to plot
        has_beats = False
        for component_name, component_data in heartbeats_dict.items():
            if component_data and "M_x" in component_data:
                beats_df = component_data["M_x"].get("mean_beat", pd.DataFrame())
                if not beats_df.empty:
                    has_beats = True
                    break
        
        if has_beats:
            heartbeats.plot_averaged_beats(
                heartbeats_dict,
                savename=os.path.join(output_dir, "4_averaged_heartbeats_overlay.pdf"),
            )
        else:
            logger.warning("No beats found for plotting - skipping averaged beats plot")


def _handle_segment_result(result, segment_idx, segment_results):
    """Handle a segment processing result."""
    if result is not None:  # Only add successful results
        segment_results.append(result)
        logger.info(f"Completed segment {segment_idx + 1}")
        return True
    else:
        logger.warning(f"Segment {segment_idx + 1} processing failed")
        return False


def _find_continuous_segments(mask, min_length=None):
    """
    Find continuous segments where mask is False (clean data).

    Parameters
    ----------
    mask : np.ndarray
        Boolean array where True indicates artifacts
    min_length : int, optional
        Minimum length of segments to keep (default: None)

    Returns
    -------
    segments : list of tuples
        List of (start_idx, end_idx) for each continuous clean segment
    """
    # Find where clean data starts and ends
    clean_data = ~mask
    diff = np.diff(np.concatenate(([False], clean_data, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = list(zip(starts, ends))

    # Filter by minimum length if specified
    if min_length is not None:
        segments = [
            (start, end) for start, end in segments if end - start >= min_length
        ]

    return segments

def _create_overlapping_windows(
    signal_length,
    window_length_samples,
    overlap_samples,
    min_window_samples=None
):
    """
    Create overlapping window indices for a signal.

    Parameters
    ----------
    signal_length : int
        Total length of signal in samples
    window_length_samples : int
        Target window size in samples
    overlap_samples : int
        Overlap between consecutive windows in samples
    min_window_samples : int, optional
        Minimum acceptable window size (default: window_length_samples // 2)

    Returns
    -------
    windows : list of tuple
        List of (start_idx, end_idx) for each window
    """
    if min_window_samples is None:
        min_window_samples = window_length_samples // 2

    step_size = window_length_samples - overlap_samples

    if step_size <= 0:
        raise ValueError(
            f"Invalid configuration: window_length ({window_length_samples}) "
            f"must be greater than overlap ({overlap_samples})"
        )

    windows = []
    current_start = 0

    while current_start < signal_length:
        current_end = min(current_start + window_length_samples, signal_length)
        window_size = current_end - current_start

        # Add window if it meets minimum size requirement
        if window_size >= min_window_samples:
            windows.append((current_start, current_end))

        # Check if we've reached the end
        if current_end >= signal_length:
            break

        # Move to next window start
        current_start += step_size

    # Handle edge case: extend last window if there's a small remainder
    if windows and windows[-1][1] < signal_length:
        remainder = signal_length - windows[-1][1]
        if remainder < min_window_samples:
            # Extend last window to cover remainder
            windows[-1] = (windows[-1][0], signal_length)

    return windows


def _windows_from_segments(
    segments,
    window_length_samples,
    overlap_samples,
    min_window_samples=None
):
    """
    Generate overlapping windows within each segment.

    Parameters
    ----------
    segments : list of tuple
        List of (start, end) segment indices
    window_length_samples : int
        Target window size in samples
    overlap_samples : int
        Overlap between consecutive windows
    min_window_samples : int, optional
        Minimum acceptable window size

    Returns
    -------
    windows : list of dict
        List of window info dictionaries:
        {
            'start': int,           # Global start index
            'end': int,             # Global end index
            'segment_idx': int,     # Which segment this window belongs to
            'local_start': int,     # Start relative to segment
            'local_end': int        # End relative to segment
        }
    """
    if min_window_samples is None:
        min_window_samples = window_length_samples // 2

    all_windows = []

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        segment_length = seg_end - seg_start

        # Skip segments smaller than minimum window size
        if segment_length < min_window_samples:
            logger.warning(
                f"Segment {seg_idx} (length={segment_length}) is smaller than "
                f"min_window_samples ({min_window_samples}), skipping"
            )
            continue

        # Generate windows for this segment
        local_windows = _create_overlapping_windows(
            segment_length,
            window_length_samples,
            overlap_samples,
            min_window_samples
        )

        # Convert to global coordinates and add metadata
        for local_start, local_end in local_windows:
            all_windows.append({
                'start': seg_start + local_start,
                'end': seg_start + local_end,
                'segment_idx': seg_idx,
                'local_start': local_start,
                'local_end': local_end
            })

    return all_windows


def _merge_overlapping_windows(
    window_results,
    total_length,
    num_dipoles=2,
    merge_method="average"
):
    """
    Merge overlapping window results by averaging.

    Parameters
    ----------
    window_results : list of dict
        Results from each window:
        {
            'start': int,
            'end': int,
            'm_hat': np.ndarray (window_len, num_dipoles, 3),
            'r_hat': np.ndarray (window_len, num_dipoles, 3),
            'segment_idx': int (optional)
        }
    total_length : int
        Total length of reconstructed signal
    num_dipoles : int
        Number of dipoles
    merge_method : str
        Merging strategy ("average")

    Returns
    -------
    m_hat : np.ndarray
        Merged dipole moments (total_length, num_dipoles, 3)
    r_hat : np.ndarray
        Merged dipole positions (total_length, num_dipoles, 3)
    overlap_counts : np.ndarray
        Number of contributing windows per sample (total_length,)
    """
    # Initialize accumulators
    m_sum = np.zeros((total_length, num_dipoles, 3), dtype=np.float64)
    r_sum = np.zeros((total_length, num_dipoles, 3), dtype=np.float64)
    counts = np.zeros(total_length, dtype=np.int32)

    # Accumulate results from each window
    for result in window_results:
        start = result['start']
        end = result['end']
        window_len = end - start

        m_hat = result['m_hat']
        r_hat = result['r_hat']

        # Validate dimensions
        if m_hat.shape[0] != window_len:
            logger.warning(
                f"Window length mismatch: expected {window_len}, got {m_hat.shape[0]}"
            )
            continue

        # Accumulate
        m_sum[start:end] += m_hat
        r_sum[start:end] += r_hat
        counts[start:end] += 1

    # Average (divide by counts, handle zeros with NaN)
    m_hat_merged = np.full_like(m_sum, np.nan)
    r_hat_merged = np.full_like(r_sum, np.nan)

    valid_mask = counts > 0
    m_hat_merged[valid_mask] = m_sum[valid_mask] / counts[valid_mask, None, None]
    r_hat_merged[valid_mask] = r_sum[valid_mask] / counts[valid_mask, None, None]

    return m_hat_merged, r_hat_merged, counts


def _process_segment_worker(args):
    """
    Worker function for parallel segment processing.
    This function needs to be at module level for multiprocessing.
    """
    try:
        segment_data, segment_time, segment_idx, config, axis_mask, W, fs_ = args
        # Use global device from config
        device = config.get("device", "cpu")

        # Import required modules within worker
        import torch
        import numpy as np
        from fmcg.utils import data, utils
        from fmcg.fitting.inverse_solver import InverseSolver

        N = segment_data.shape[0]
        y = torch.tensor(segment_data, dtype=torch.float32, device=device)

        # Initialize solver and parameters
        r_init = np.repeat(config["solver"]["r_init"], N, axis=0)
        mdl = _create_inverse_solver(config, N, axis_mask, W)
        initial_parameters, field_true = _initialize_solver_parameters(
            mdl, r_init, y, config
        )

        # Solve for this segment
        m_hat, r_hat = _solve_segment(mdl, field_true, initial_parameters, config, segment_idx=segment_idx)

        # Apply wavelet denoising
        m_hat = _apply_wavelet_denoising(m_hat, config)

        # Convert to CPU/numpy before returning (important for multiprocessing)
        result = {
            "m_hat": m_hat.cpu().numpy() if hasattr(m_hat, "cpu") else m_hat,
            "r_hat": r_hat.cpu().numpy() if hasattr(r_hat, "cpu") else r_hat,
            "best_loss": (
                mdl.best_loss.tolist()
                if hasattr(mdl.best_loss, "tolist")
                else float(mdl.best_loss)
            ),
            "time": segment_time,
            "segment_idx": segment_idx,
        }

        # Clean up GPU memory in this worker
        if device.startswith("cuda"):
            try:
                del y, m_hat, r_hat, mdl, initial_parameters, field_true, r_init
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")

        return result

    except Exception as e:
        logger.error(f"Error processing segment {segment_idx}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None  # Return None for failed segments


def _process_segment_worker_threaded(
    pipeline_ref, segment_data, segment_time, segment_idx
):
    """
    Worker function for threaded parallel segment processing.
    Uses threading for CUDA compatibility - all threads share the same device.
    """
    try:
        N = segment_data.shape[0]
        # Use the device from pipeline config
        device = pipeline_ref.config.get("device", "cpu")
        y = torch.tensor(segment_data, dtype=torch.float32, device=device)

        # Initialize solver and parameters
        r_init = np.repeat(pipeline_ref.config["solver"]["r_init"], N, axis=0)
        mdl = _create_inverse_solver(
            pipeline_ref.config, N, pipeline_ref.axis_mask, pipeline_ref.W
        )
        initial_parameters, field_true = _initialize_solver_parameters(
            mdl, r_init, y, pipeline_ref.config
        )

        # Solve for this segment
        m_hat, r_hat = _solve_segment(
            mdl, field_true, initial_parameters, pipeline_ref.config, segment_idx=segment_idx
        )

        # Apply wavelet denoising
        m_hat = _apply_wavelet_denoising(m_hat, pipeline_ref.config)

        result = {
            "m_hat": m_hat.cpu().numpy() if hasattr(m_hat, "cpu") else m_hat,
            "r_hat": r_hat.cpu().numpy() if hasattr(r_hat, "cpu") else r_hat,
            "mdl": mdl,
            "best_loss": (
                mdl.best_loss.cpu().item()
                if hasattr(mdl.best_loss, "cpu")
                else float(mdl.best_loss)
            ),
            "best_i": mdl.best_i,
            "time": segment_time,
            "segment_idx": segment_idx,
        }

        return result

    except Exception as e:
        logger.error(f"Error processing segment {segment_idx}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
