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


def _create_basis_dict(config, N):
    """Create basis dictionary for solver configuration."""
    basis_dict = {}
    
    # Handle r basis - support both "r" and "r_basis" keys
    r_config = config["solver"].get("r") or config["solver"].get("r_basis")
    if r_config is not None:
        if isinstance(r_config, dict):
            basis_dict["r"] = dict(r_config)  # shallow copy
            basis_dict["r"]["n_time"] = N
        else:
            basis_dict["r"] = {"basis": r_config, "n_time": N}
        # Add r_bnds if specified separately (legacy support)
        if "r_bnds" in config["solver"]:
            basis_dict["r"]["bnds"] = config["solver"]["r_bnds"]
    
    # Handle m basis - support both "m" and "m_basis" keys
    m_config = config["solver"].get("m") or config["solver"].get("m_basis")
    if m_config is not None:
        if isinstance(m_config, dict):
            basis_dict["m"] = dict(m_config)  # shallow copy
            basis_dict["m"]["n_time"] = N
        else:
            basis_dict["m"] = {"basis": m_config, "n_time": N}
    
    return basis_dict

def _create_inverse_solver(config, N, axis_mask, W, r_sensors, loss_mask=None):
    """Create and initialize InverseSolver instance.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    N : int
        Number of time steps.
    axis_mask : array-like
        Valid-axis mask (S, 3).
    W : array-like or None
        Whitening matrix.
    r_sensors : array-like
        Sensor positions (S, 3).
    loss_mask : array-like or None
        Optional mask (S, 3) for excluding sensors from the loss only
        (e.g. for cross-validation).  Unlike axis_mask, this does NOT
        affect the forward model or whitening dimensions.
    """

    # Determine device once from config
    device = config.get("device", "cpu")

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
        loss_mask=loss_mask,
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
        seed=config.get("seed", None),
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


def _process_segment_worker(args):
    """
    Worker function for parallel segment processing.
    This function needs to be at module level for multiprocessing.
    """
    try:
        segment_data, segment_time, segment_idx, config, axis_mask, W, r_sensors, fs_ = args
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
        mdl = _create_inverse_solver(config, N, axis_mask, W, r_sensors)
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
            "best_i": mdl.best_i,
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
            pipeline_ref.config, N, pipeline_ref.axis_mask, pipeline_ref.W, pipeline_ref.r_sensors
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


def _compute_hrv_quality_metrics(data_dict, component, fs, config):
    """Compute HRV quality metrics for one component from final detected peaks.

    Reports raw (all peaks) and clean (outlier-filtered) variants of sdnn and rmssd.
    An IBI is considered clean if neither of its two bounding peaks is flagged by
    detect_hr_outlier. Outlier detection parameters are read from
    config["post_processing"]["averaging"]["outlier_kwargs"], matching the settings
    used during heartbeat processing.

    Returns a dict with keys:
        sdnn_{component}            std of NN intervals (ms, all peaks)
        sdnn_clean_{component}      std of NN intervals (ms, non-outlier IBIs only)
        rmssd_{component}           sqrt(mean(successive NN diffs^2)) (ms, all peaks)
        rmssd_clean_{component}     sqrt(mean(successive NN diffs^2)) (ms, valid IBIs only)
        outlier_rate_{component}    fraction of peaks flagged as HR outliers
        num_valid_peaks_{component} count of non-outlier peaks
    """
    suffix = f"_{component}"
    nan_result = {
        f"sdnn{suffix}": float("nan"),
        f"sdnn_clean{suffix}": float("nan"),
        f"rmssd{suffix}": float("nan"),
        f"rmssd_clean{suffix}": float("nan"),
        f"outlier_rate{suffix}": float("nan"),
        f"num_valid_peaks{suffix}": 0,
    }

    try:
        comp_data = data_dict.get(component, {})
        peaks = comp_data.get("peaks", None)
        if peaks is None or len(peaks) < 3:
            return nan_result

        outlier_kwargs = config["post_processing"]["averaging"].get(
            "outlier_kwargs", {"win": 4, "threshold": 10}
        )

        ibi_ms = np.diff(peaks) / fs * 1000  # inter-beat intervals in ms, length N-1

        # --- Raw metrics (all peaks) ---
        sdnn = float(np.std(ibi_ms))
        rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))

        # --- Outlier detection (same parameters as heartbeat processing) ---
        outlier = heartbeats.detect_hr_outlier(
            peaks, fs,
            win=outlier_kwargs.get("win", 4),
            threshold=outlier_kwargs.get("threshold", 10),
            plot=False,
        )
        outlier_rate = float(np.mean(outlier)) if len(outlier) > 0 else float("nan")
        num_valid_peaks = int(np.sum(~outlier))

        # --- Clean metrics (IBIs where both bounding peaks are non-outliers) ---
        valid_ibi_mask = ~outlier[:-1] & ~outlier[1:]  # length N-1
        valid_ibis = ibi_ms[valid_ibi_mask]

        sdnn_clean = float(np.std(valid_ibis)) if len(valid_ibis) > 1 else float("nan")
        rmssd_clean = (
            float(np.sqrt(np.mean(np.diff(valid_ibis) ** 2)))
            if len(valid_ibis) > 2
            else float("nan")
        )

        return {
            f"sdnn{suffix}": round(sdnn, 2),
            f"sdnn_clean{suffix}": round(sdnn_clean, 2) if not np.isnan(sdnn_clean) else float("nan"),
            f"rmssd{suffix}": round(rmssd, 2),
            f"rmssd_clean{suffix}": round(rmssd_clean, 2) if not np.isnan(rmssd_clean) else float("nan"),
            f"outlier_rate{suffix}": round(outlier_rate, 4),
            f"num_valid_peaks{suffix}": num_valid_peaks,
        }

    except Exception as e:
        logger.warning(f"HRV quality metrics failed for {component}: {e}")
        return nan_result
