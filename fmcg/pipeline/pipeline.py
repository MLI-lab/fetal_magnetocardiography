"""Provides methods to post-process and analyze fetal magnetocardiography (fMCG) data."""

import ast
import csv
import os
import datetime
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing
import threading
import traceback
import logging
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.stats
import torch
import neurokit2 as nk
import dataclasses

from fmcg.utils.plotting.report import report_hr, report_ica_components, report_m_hat_segments, report_r_hat_segments

from ..utils import data, utils
from ..utils.plotting import plot_utils
from fmcg.signal.whitening import apply_whitening
from fmcg.signal.artifact_removal import detect_artifacts
from fmcg.signal.filtering import filter_sensor_dict
from fmcg.signal.artifact_removal import remove_outlier_dict
from ..fitting.field_model import ForwardModel


from ._pipeline_utils import (
    _create_inverse_solver,
    _initialize_solver_parameters,
    _plot_averaged_beats_if_enabled,
    _process_dipole_data,
    _segment_and_average_beats,
    _solve_segment,
    _apply_wavelet_denoising,
    _find_continuous_segments,
    _process_segment_worker,
    _process_segment_worker_threaded,
    _handle_segment_result,
)


np.set_printoptions(legacy="1.21")
logger = logging.getLogger(__name__)


# Limit numpy/BLAS threading to prevent oversubscription in post-processing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


class Pipeline:
    """
    This class encapsulates the complete workflow for fMCG data analysis, including data loading,
    preprocessing, solver initialization, optimization, post-processing, and reporting. It is designed
    to be modular and configurable, allowing for flexible adaptation to different datasets and analysis
    requirements.

    config : dict
        Configuration dictionary specifying pipeline parameters, solver settings, preprocessing options,
        and output preferences.
    system_config : dict
        System configuration for the measurement device and acquisition setup.
    measurement_config : dict
        Measurement configuration describing the data structure and acquisition details.
    mask : np.ndarray or None, optional
        Optional mask to exclude specific sensor axes from analysis.
    log_note : str, optional
        Optional note to include in the log record.
    log_dict : dict, optional
        Optional dictionary for logging additional information.
    verbose : bool, optional
        If True, enables verbose logging.

    Attributes
    sig_data_dict_raw : dict
        Raw signal data loaded from the dataset.
    noise_data_dict_raw : dict
        Raw noise data loaded from the dataset.
    sensor_dict : dict
        Preprocessed signal data after filtering and outlier removal.
    noise_sensor_dict : dict
        Preprocessed noise data after filtering and outlier removal.
    field_maps : np.ndarray
        Array of filtered and optionally whitened signal data.
    axis_mask : np.ndarray
        Mask indicating valid sensor axes after outlier removal.
    clean_segments : list of tuple
        List of (start, end) indices for clean data segments (if segmented processing is enabled).
    segment_results : list of dict
        Results from solver applied to each segment (if segmented processing is enabled).
    data_dict : dict
        Processed dipole moment data and extracted heartbeat information.
    heartbeats_dict : dict
        Averaged heartbeat data and segmentation results.
    output_dir : str
        Directory for saving outputs, reports, and logs.
    log_dict : dict
        Dictionary containing pipeline execution logs and metrics.
    log_note : str
        Note describing the current pipeline run.

    Methods
    run(config, system_config, measurement_config, mask=None, log_note="", log_dict={}, verbose=False)
        Run the complete pipeline with the specified configuration and parameters.
    load_measurements_from_labels(labels_path, measurement_config_date=None, sorted=True)
        Load measurement data based on specified labels and configuration.

    Notes
    -----
    - If no noise data is provided in the configuration, the pipeline will not execute and will return early.
    - Segmented processing splits the data into continuous clean segments based on artifacts detection.
    - Parallel processing is available for segment-wise solver execution.
    - All outputs, logs, and reports are saved to the specified output directory.

    Examples
    --------
    >>> pipeline = Pipeline()
    >>> pipeline.run(config, system_config, measurement_config, mask=my_mask, log_note="Trial 1", verbose=True)
    """

    def __init__(self):
        self._step_num = 1
        self._device_context = None

    def __enter__(self):
        """Enter context manager - setup resources."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup resources."""
        # Clean up GPU memory if using CUDA
        if hasattr(self, 'device') and 'cuda' in self.device:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug(f"Cleared CUDA cache for device {self.device}")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")

        # Return False to propagate exceptions
        return False

    def _log_step_header(self, description):
        """Helper method to log consistent step headers."""
        logger.info(f"\n{'='*40}")
        logger.info(f"Step {self._step_num}: {description}")
        logger.info(f"{'='*40}")
        self._step_num += 1

    def _create_solver_components(self, data, device=None):
        """
        Create solver components (model, initial parameters, field_true).
        Used for both segmented and non-segmented processing.
        """
        N = data.shape[0]
        r_init = np.repeat(self.config["solver"]["r_init"], N, axis=0)
        
        mdl = _create_inverse_solver(self.config, N, self.axis_mask, self.W)
        
        if device:
            y = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            y = data  # Assume it's already a tensor or numpy array
            
        initial_parameters, field_true = _initialize_solver_parameters(
            mdl, r_init, y, self.config
        )
        
        return mdl, initial_parameters, field_true

    def _plot_sensor_signals_if_enabled(self, data, time, ylabel, filename_suffix, xlim=None, **kwargs):
        """Helper method to plot sensor signals only if plotting is enabled."""
        if not self.save_plots:
            return
            
        default_kwargs = {
            'xlim': [50, 52] if xlim is None else xlim,
            'sharey': True,
            'sensor_names': list(self.sensor_dict.keys())
        }
        default_kwargs.update(kwargs)
        
        plot_utils.plot_sensor_signals(
            data,
            time,
            ylabel=ylabel,
            savename=os.path.join(self.output_dir, filename_suffix),
            **default_kwargs
        )

    def _execute_common_post_processing(self):
        """
        Common post-processing logic extracted to eliminate duplication
        between segmented and non-segmented processing.
        """
        # Process the dipole moments
        # Pass artifact mask if available (for overlapping windows)
        artifacts_mask = getattr(self, "artifacts_mask", None)
        self.data_dict = _process_dipole_data(
            self.m_hat, self.r_hat, self.fs_, self.config, 
            self.log_dict, False, artifacts_mask=artifacts_mask
        )
        
        # Segment and average heartbeats
        self.heartbeats_dict = _segment_and_average_beats(
            self.data_dict, self.fs_, self.config, self.log_dict
        )
        
        # Log heartbeat findings
        for k in self.heartbeats_dict.keys():
            if self.log_dict and f"selected_beats_{k}" in self.log_dict and f"selected_hr_{k}" in self.log_dict:
                logger.info(f"Found {k} {self.log_dict[f'selected_beats_{k}']:.2f} heartbeats with mean HR {self.log_dict[f'selected_hr_{k}']:.2f} bpm")

        # Compute goodness-of-fit metrics (reconstruction quality, residual stats, discrepancy)
        self._compute_gof_metrics()

        # Save data if required - unified pickle file format
        if self.save_data:
            save_dict = {
                "data_dict": self.data_dict,
                "heartbeats_dict": self.heartbeats_dict,
            }
            # Include ground truth if available (synthetic data)
            if hasattr(self, 'ground_truth') and self.ground_truth:
                save_dict["ground_truth"] = self.ground_truth
                logger.info("Including ground truth in pipeline_results.pkl")

            with open(os.path.join(self.output_dir, "pipeline_results.pkl"), "wb") as f:
                pickle.dump(save_dict, f)
        
        # Plot results
        _plot_averaged_beats_if_enabled(
            self.heartbeats_dict, self.save_plots, self.output_dir
        )

    def _compute_gof_metrics_for(self, field_windowed, m_hat, r_hat):
        """Compute GOF metrics for one field/m_hat/r_hat triplet.

        Returns a dict of scalar metrics, or an empty dict if computation is not possible.
        All metrics are computed in the whitened domain.
        """
        r_sensors = data.generate_array_coordinates(
            grid_shape=(4, 4), grid_spacing=0.04, y=0
        )
        fm = ForwardModel(r_sensors=r_sensors, device="cpu")

        if field_windowed.shape[0] != m_hat.shape[0]:
            logger.warning(
                f"GOF metrics skipped: field length ({field_windowed.shape[0]}) "
                f"!= m_hat length ({m_hat.shape[0]})"
            )
            return {}

        # Valid timepoints: no NaN in dipole solution
        valid_t = ~np.isnan(m_hat.reshape(len(m_hat), -1)).any(axis=1)
        if valid_t.sum() < 10:
            logger.warning("GOF metrics skipped: fewer than 10 valid timepoints")
            return {}

        m_valid = m_hat[valid_t]
        r_valid = r_hat[valid_t]
        field_valid = field_windowed[valid_t]

        # Forward model prediction in physical units
        r_tensor = torch.tensor(r_valid, dtype=torch.float32)
        m_tensor = torch.tensor(m_valid, dtype=torch.float32)
        field_pred_phys = fm.forward_linear(r_tensor, m_tensor, as_numpy=True, silent=True)
        # field_pred_phys: (T_valid, S, 3)

        # Extract valid channels
        valid_ch_mask = self.axis_mask == 1          # (S, 3) bool
        field_meas_w = field_valid[:, valid_ch_mask]  # (T_valid, n_valid)
        field_pred_ch = field_pred_phys[:, valid_ch_mask]  # (T_valid, n_valid)

        # Apply whitening to prediction to match the measured whitened field
        if self.W is not None:
            field_pred_w = field_pred_ch @ self.W.T
        else:
            field_pred_w = field_pred_ch

        # Remove channels with NaN in the measured field
        valid_ch = ~np.isnan(field_meas_w).any(axis=0)
        field_meas_w = field_meas_w[:, valid_ch]
        field_pred_w = field_pred_w[:, valid_ch]

        residual = field_meas_w - field_pred_w  # (T_valid, n_ch)

        # --- Group 1: Reconstruction quality ---
        norm_res = np.linalg.norm(residual)
        norm_meas = np.linalg.norm(field_meas_w)
        relative_error = norm_res / (norm_meas + 1e-12)
        r_squared = 1.0 - relative_error ** 2
        srr_db = 20.0 * np.log10(np.linalg.norm(field_pred_w) / (norm_res + 1e-12))

        # --- Group 2: Residual distribution statistics ---
        res_flat = residual.ravel()
        metrics = {
            "recon_r_squared": round(float(r_squared), 4),
            "recon_relative_error_pct": round(float(relative_error * 100), 4),
            "recon_srr_db": round(float(srr_db), 4),
            "recon_rms_residual": round(float(np.sqrt(np.mean(residual ** 2))), 6),
            "residual_mean": round(float(np.mean(res_flat)), 6),
            "residual_std": round(float(np.std(res_flat)), 6),
            "residual_skewness": round(float(scipy.stats.skew(res_flat)), 4),
            "residual_kurtosis": round(float(scipy.stats.kurtosis(res_flat)), 4),
        }

        # --- Group 3: Discrepancy threshold (whitened domain) ---
        if self.noise_whitened is not None:
            noise_w = self.noise_whitened[:, valid_ch]  # (T_noise, n_ch)
            noise_norm_sq = np.sum(noise_w ** 2, axis=1)
            field_norm_sq = np.sum(field_meas_w ** 2, axis=1)
            residual_norm_sq = np.sum(residual ** 2, axis=1)
            discrepancy_threshold = 0.5 * np.mean(noise_norm_sq) / (np.mean(field_norm_sq) + 1e-12)
            data_fit = 0.5 * np.mean(residual_norm_sq) / (np.mean(field_norm_sq) + 1e-12)
            metrics.update({
                "discrepancy_threshold": round(float(discrepancy_threshold), 6),
                "data_fit": round(float(data_fit), 6),
                "fit_within_noise": int(data_fit < discrepancy_threshold),
            })

        return metrics

    def _compute_gof_metrics(self):
        """Non-segmented path: compute GOF on the full window, store scalars to self.log_dict."""
        metrics = self._compute_gof_metrics_for(
            self.field_maps_windowed, self.m_hat, self.r_hat
        )
        if metrics:
            self.log_dict.update(metrics)
            logger.info(
                f"GOF: R²={metrics['recon_r_squared']}, "
                f"SRR={metrics['recon_srr_db']} dB, "
                f"residual_std={metrics['residual_std']}"
            )

    def _initialize_output_directory(self):
        # Create output directory if it does not exist
        if self.save_reports or self.save_plots or self.save_data:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = (
                f"{self.config['data']['patient']}_{self.config['data']['series']}_{self.config['data']['sig_group_names']}_{self.config['data']['sampfrom']}s-{self.config['data']['sampto']}s_{dt_str}"
                + (f" - {self.log_note}" if self.log_note else "")
            )
            self.output_dir = os.path.join(self.output_dir, output_folder_name)
            os.makedirs(self.output_dir, exist_ok=True)

    def _load_data(self):
        """
        Load the fMCG data and noise data from the specified dataset path.
        If data_loader is provided (synthetic mode), use it instead of TDMS loading.
        """

        self._log_step_header("Data Loading")
        
        if self.data_loader is not None:
            # Synthetic data mode
            logger.info("Loading synthetic data...")
            self.sig_data_dict_raw, self.time, self.fs, self.noise_data_dict_raw = (
                self.data_loader.load_structured_patient_data_and_noise(
                    self.systemconfig,
                    self.measurementconfig,
                )
            )
            # Store ground truth for evaluation
            self.ground_truth = self.data_loader.ground_truth
            logger.info("Synthetic data loaded with ground truth")
        else:
            # Normal TDMS mode
            logger.info(f"Dataset Path: {self.config['data']['ds_path']}")
            logger.info(
                f"Patient: {self.config['data']['patient']}, Series: {self.config['data']['series']}"
            )

            # Load the raw signal and noise data
            self.sig_data_dict_raw, self.time, self.fs, self.noise_data_dict_raw = (
                data.load_structured_patient_data_and_noise(
                    self.systemconfig,
                    self.measurementconfig,
                    **{
                        k: self.config["data"][k]
                        for k in self.config["data"].keys()
                        & {
                            "ds_path",
                            "patient",
                            "series",
                            "sig_group_names",
                            "noise_group_names",
                            "noise_patient",
                            "noise_series",
                        }
                    },
                    files="all",
                )
            )
            
        signal_length = len(list(self.sig_data_dict_raw.values())[0]) / self.fs
        noise_length = len(list(self.noise_data_dict_raw.values())[0]) / self.fs

        logger.info(f"Loaded signal ({signal_length:.2f} s) and noise ({noise_length:.2f} s)")

        self.log_dict["signal_length"] = round(signal_length, 2)
        self.log_dict["noise_length"] = round(noise_length, 2)

    def _preprocess_data(self):
        """
        Preprocess the loaded fMCG data by applying a bandpass filter, removing outliers, and optionally whitening the data.
        """
        self._log_step_header("Data Preprocessing")
        bandpass_low = self.config["preprocessing"]["bandpass_low"]
        bandpass_high = self.config["preprocessing"]["bandpass_high"]
        logger.info(f"Bandpass Filter: {bandpass_low} - {bandpass_high} Hz")

        # Apply bandpass and notch filter
        bandpass_order = self.config["preprocessing"]["bandpass_order"]
        self.sensor_dict = filter_sensor_dict(
            self.sig_data_dict_raw,
            self.fs,
            low=bandpass_low,
            high=bandpass_high,
            order=bandpass_order,
        )

        self.noise_sensor_dict = filter_sensor_dict(
            self.noise_data_dict_raw,
            self.fs,
            low=bandpass_low,
            high=bandpass_high,
            order=bandpass_order,
        )

        # # Plot short excerpt as sanity check
        # if self.save_plots:
        #     utils.sanity_check(
        #         self.sensor_dict,
        #         self.time,
        #         savename=os.path.join(self.output_dir, "0_sanitycheck.pdf"),
        #     )

        # Remove outliers based on physical range thresholds
        logger.info("\nSignal Recording:")
        signal_threshold = self.config["preprocessing"]["signal_threshold"]
        self.sensor_dict, axis_mask_signal = remove_outlier_dict(
            self.sensor_dict,
            physical_range_threshold=signal_threshold,
        )

        logger.info("\nNoise Recording:")
        noise_threshold = self.config["preprocessing"]["noise_threshold"]
        self.noise_sensor_dict, axis_mask_noise = remove_outlier_dict(
            self.noise_sensor_dict,
            physical_range_threshold=noise_threshold,
        )
        
        # Combine axis masks for signal and noise, removing axes that are outliers
        if axis_mask_signal.shape != axis_mask_noise.shape:
            raise ValueError(f"Axis mask shapes do not match: signal {axis_mask_signal.shape}, noise {axis_mask_noise.shape}")
        if axis_mask_signal.dtype != axis_mask_noise.dtype:
            axis_mask_signal = axis_mask_signal.astype(np.int32)
            axis_mask_noise = axis_mask_noise.astype(np.int32)
        self.axis_mask = axis_mask_signal & axis_mask_noise

        # Optionally masking some axes
        if self.mask is not None:
            self.axis_mask = self.axis_mask & self.mask

        # Convert sensor dictionaries to numpy arrays for further processing
        field_maps = np.array([v for k, v in self.sensor_dict.items()]).transpose(
            1, 0, 2
        )
        noise_maps = np.array([v for k, v in self.noise_sensor_dict.items()]).transpose(
            1, 0, 2
        )

        # Plot filtered signals and noise
        self._plot_sensor_signals_if_enabled(
            field_maps,
            self.time,
            ylabel="Filtered Field [pT]",
            filename_suffix="1_signal_frequencyfiltered.pdf"
        )
        self._plot_sensor_signals_if_enabled(
            noise_maps,
            np.arange(0, len(noise_maps)) / self.fs,
            ylabel="Filtered Noise [pT]",
            filename_suffix="1_noise_frequencyfiltered.pdf",
            sensor_names=list(self.noise_sensor_dict.keys())
        )

        # Detect (temporal) artifacts in the signal
        detect_artifacts_enabled = self.config["preprocessing"].get("detect_artifacts", True)
        if detect_artifacts_enabled:
            artifacts_mask = detect_artifacts(
                field_maps[:, self.axis_mask == 1],
                fs=self.fs,
                wavelet="haar",
                level=7,
                segment_length_seconds=10,
                struct_time=10,
                channel_agreement_percentage=0.05,
                coeff_depth=None,
                offset=1,
                n_sigma=3,
            )
        else:
            logger.info("Artifact detection disabled, using empty artifacts mask")
            artifacts_mask = np.zeros(field_maps.shape[0], dtype=bool)
        self.artifacts_mask = artifacts_mask



        if self.save_plots:
            # Plot artifacts mask
            plt.figure(figsize=(10, 5))
            plt.plot(self.time, artifacts_mask, label="Artifacts Mask", color="red")
            plt.xlabel("Time [s]")
            plt.ylabel("Mask")
            plt.savefig(os.path.join(self.output_dir, "1_artifacts_mask.pdf"))
            plt.close()

            # plot artifacts the whole signal and highlight artifacts
            self._plot_sensor_signals_if_enabled(
                field_maps,
                self.time,
                ylabel="Field [pT]",
                filename_suffix="1_signal_artifacts.pdf",
                artifacts_mask=artifacts_mask,
                xlim=[0, self.time[-1]],
            )

        # Apply the artifacts mask to the field maps and noise maps
        # field_maps[artifacts_mask] = 0

        # compute signal power and round to 4 decimals
        self.sig_power = round(
            utils.sig_power(field_maps[~artifacts_mask], root=True, win=10 * self.fs), 4
        )
        self.noise_power = round(
            utils.sig_power(noise_maps, root=True, win=10 * self.fs), 4
        )

        # whitening
        whitening_method = self.config["preprocessing"].get("whitening")
        if whitening_method is not None:
            logger.info(f"\nApplying {whitening_method} Whitening")

            rescale_whitening = self.config["preprocessing"].get("rescale_whitening")
            self.field_maps, self.W = apply_whitening(
                field_maps[~artifacts_mask],
                noise_maps,
                self.axis_mask,
                method=whitening_method,
                return_whitening_matrix=True,
                rescale=rescale_whitening,
            )
            # Store whitened noise for GOF discrepancy threshold
            noise_valid = noise_maps[:, self.axis_mask == 1].copy()
            noise_valid -= np.nanmean(noise_valid, axis=0)
            self.noise_whitened = noise_valid @ self.W.T
        else:
            self.field_maps = field_maps
            self.W = None
            self.noise_whitened = None

        field_maps[~artifacts_mask] = self.field_maps
        self.field_maps = field_maps

        # Plot the whitened signals
        if self.config["preprocessing"].get("whitening") is not None:
            self._plot_sensor_signals_if_enabled(
                self.field_maps,
                self.time,
                ylabel="Whitened Field [pT]",
                filename_suffix="1_signal_whitened.pdf"
            )

        # Split the data into clean segmentents by removing artifacts
        if self.config["preprocessing"].get("process_segments", False):
            # Find continuous clean segments
            min_segment_length = int(
                self.config["preprocessing"].get("min_segment_length", 30) * self.fs
            )  # Default 30s
            self.clean_segments = _find_continuous_segments(
                artifacts_mask, min_length=min_segment_length
            )
            
            if len(self.clean_segments) == 0:
                print(f"WARNING: No clean segments found! Check min_segment_length and artifact pattern.")

            total_clean_time = (
                sum(end - start for start, end in self.clean_segments) / self.fs
            )
            logger.info(
                f"\nArtifact Removal:\nFound {len(self.clean_segments)} clean segments with total duration: {total_clean_time:.2f}s"
            )
            for i, (start, end) in enumerate(self.clean_segments):
                logger.info(
                    f"  Segment {i+1}: {start/self.fs:.1f}s - {end/self.fs:.1f}s ({(end-start)/self.fs:.1f}s)"
                )

            self.log_note += f"Found {len(self.clean_segments)} clean segments. "
        else:
            # Original behavior: remove artifacts and concatenate
            self.field_maps = self.field_maps[~artifacts_mask]
            self.time = self.time[~artifacts_mask]
            self.log_note += f"Removed {np.sum(artifacts_mask)/self.fs:.2f} s artifacts from the data. "

    def _decimate_data(self):
        """
        Downsample the fMCG data by a specified factor to reduce computational load.
        """
        q = self.config["preprocessing"]["downsample_factor"]
        logger.info(f"Downsampling by factor {q}")

        self.fs_ = int(self.fs / q)
        # For segmented processing, store original data and decimated segments
        self.field_maps_dec = scipy.signal.decimate(self.field_maps, q, axis=0)
        # self.time_dec = scipy.signal.decimate(self.time, q, axis=0)
        # self.field_maps_dec = scipy.signal.resample_poly(self.field_maps, up=1, down=q, axis=0)
        self.time_dec = np.linspace(self.time[0], self.time[-1], num=self.field_maps_dec.shape[0], endpoint=True)

        if hasattr(self, "clean_segments"):
            # Update segment indices for decimated data
            self.clean_segments_dec = [
                (start // q, end // q) for start, end in self.clean_segments
            ]

        # CRITICAL: Decimate ground truth to match decimated field data
        if hasattr(self, 'ground_truth') and self.ground_truth:
            logger.info("Decimating ground truth arrays to match downsampled data")
            gt = self.ground_truth

            # Decimate m_true, r_true, time, and field_clean if present
            for key in ['m_true', 'r_true', 'field_clean', 'time']:
                if key in gt and isinstance(gt[key], np.ndarray):
                    original_shape = gt[key].shape
                    # Decimate along time axis (axis 0)
                    gt[key] = scipy.signal.decimate(gt[key], q, axis=0)
                    logger.info(f"  {key}: {original_shape} -> {gt[key].shape}")

            # Update sampling rate
            if 'fs' in gt:
                gt['fs'] = self.fs_
                logger.info(f"  fs: {self.fs} -> {gt['fs']} Hz")

    def _select_time_window(self):
        """Selecting the time window to process."""
        logger.info(
            f"Processing Time Range: {self.config['data']['sampfrom']}s to {self.config['data']['sampto']}s"
        )

        # For segmented processing, apply time window to each segment
        t0 = int(self.config["data"]["sampfrom"] * self.fs_)
        sampto = self.config["data"].get("sampto")
        if sampto is None:
            t1 = len(self.field_maps_dec)
        else:
            t1 = int(sampto * self.fs_)

        if hasattr(self, "clean_segments_dec"):
            # Filter segments to be within the time window and adjust indices
            self.processing_segments = []
            for start, end in self.clean_segments_dec:
                # Clip segment to time window
                seg_start = max(start, t0)
                seg_end = min(end, t1)

                if seg_start < seg_end:  # Only keep non-empty segments
                    self.processing_segments.append((seg_start - t0, seg_end - t0))

            # Extract data for time window
            logger.info(
                f"After time windowing: {len(self.processing_segments)} segments remain"
            )
            self.log_dict["segments"] = [
                round((seg[1] - seg[0]) / self.fs_, 2)
                for seg in self.processing_segments
            ]
        else:
            # non-segmented processing
            self.log_dict["segments"] = [round((t1 - t0) / self.fs_, 2)]
        self.field_maps_windowed = self.field_maps_dec[t0:t1].copy()
        self.time_windowed = self.time_dec[t0:t1]

    def _process_single_segment(self, segment_data, segment_time, segment_idx=0):
        """
        Process a single segment of data through the solver only.
        Heartbeat segmentation and averaging will be done on combined data.

        Parameters
        ----------
        segment_data : np.ndarray
            Field map data for this segment
        segment_time : np.ndarray
            Time array for this segment
        segment_idx : int
            Index of the segment being processed

        Returns
        -------
        dict
            Dictionary containing solver results for this segment
        """
        # Log step header only for the first segment to avoid redundancy
        if segment_idx == 0:
            self._log_step_header("Solver Initialization & Solving (Segmented)")
        
        logger.info(
            f"\n--- Processing Segment {segment_idx + 1} ({len(segment_data)/self.fs_:.1f}s) ---"
        )

        # Initialize solver and parameters using helper method
        mdl, initial_parameters, field_true = self._create_solver_components(
            segment_data, device=self.device
        )

        # Solve for this segment
        m_hat, r_hat = _solve_segment(mdl, field_true, initial_parameters, self.config, segment_idx)

        # Apply wavelet denoising
        m_hat = _apply_wavelet_denoising(m_hat, self.config)

        return {
            "m_hat": m_hat,
            "r_hat": r_hat,
            "mdl": mdl,
            "best_loss": mdl.best_loss.tolist(),
            "best_i": mdl.best_i,
            "time": segment_time,
            "segment_idx": segment_idx,
        }

    def _initialize_solver(self):
        """Initialize the inverse solver with the specified parameters and initial conditions."""
        if hasattr(self, "processing_segments"):
            # For segmented processing, solver initialization happens per segment
            # No initialization needed here - just log the configuration
            logger.info(f"Number of Dipoles: {self.config['solver']['num_dipoles']}")
            logger.info(
                "Using segmented processing - solvers will be initialized per segment"
            )
            return

        # Original behavior for non-segmented processing
        self._log_step_header("Solver Initialization")
        logger.info(f"Number of Dipoles: {self.config['solver']['num_dipoles']}")

        self.mdl, self.initial_parameters, self.field_true = self._create_solver_components(
            self.field_maps_windowed
        )

    def _solve(self):
        """Solve the inverse problem using the initialized solver and the preprocessed data."""
        if hasattr(self, "processing_segments"):
            # For segmented processing, solve each segment individually
            # Step header logging is now handled in _process_single_segment
            logger.info(f"Processing {len(self.processing_segments)} segments")

            # Check if parallel processing is enabled
            use_parallel = self.config["solver"].get("parallel_processing", False)
            max_workers = self.config["solver"].get("max_workers", min(4, len(self.processing_segments)))
            max_workers = min(
                len(self.processing_segments), max_workers
            )  # Ensure at least 1 worker

            if use_parallel and len(self.processing_segments) > 1:
                # Add step header for parallel processing
                self._log_step_header("Solver Initialization & Solving (Segmented)")
                
                # Determine whether to use threading (for CUDA) or multiprocessing (for CPU)
                use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

                if use_cuda:
                    # Use threading for CUDA (shared memory/context)
                    logger.info(
                        f"Using threaded parallel processing with {max_workers} workers on {self.device}"
                    )

                    # Clear CUDA cache before starting
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # Process segments using threading
                    self.segment_results = []
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for i, (start, end) in enumerate(self.processing_segments):
                            segment_data = self.field_maps_windowed[start:end]
                            segment_time = self.time_windowed[start:end]

                            future = executor.submit(
                                _process_segment_worker_threaded,
                                self,
                                segment_data,
                                segment_time,
                                i,
                            )
                            futures.append((future, i))

                        for future, segment_idx in futures:
                            try:
                                result = future.result()
                                _handle_segment_result(
                                    result, segment_idx, self.segment_results
                                )
                            except Exception as exc:
                                logger.error(
                                    f"Segment {segment_idx + 1} generated an exception: {exc}"
                                )
                                logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    # Use multiprocessing for CPU
                    logger.info(
                        f"Using multiprocess parallel processing with {max_workers} workers on CPU"
                    )

                    # Prepare arguments for parallel processing
                    segment_args = []
                    for i, (start, end) in enumerate(self.processing_segments):
                        segment_data = self.field_maps_windowed[start:end]
                        segment_time = self.time_windowed[start:end]
                        # Use global device from config in worker
                        args = (
                            segment_data,
                            segment_time,
                            i,
                            self.config,
                            self.axis_mask,
                            self.W,
                            self.fs_,
                        )
                        segment_args.append(args)

                    # Process segments in parallel
                    self.segment_results = []
                    
                    # Use spawn context to avoid fork issues
                    mp_context = multiprocessing.get_context('spawn')
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                        future_to_idx = {
                            executor.submit(_process_segment_worker, args): args[2]
                            for args in segment_args
                        }

                        for future in as_completed(future_to_idx):
                            segment_idx = future_to_idx[future]
                            try:
                                result = future.result()
                                _handle_segment_result(
                                    result, segment_idx, self.segment_results
                                )
                            except Exception as exc:
                                logger.error(
                                    f"Segment {segment_idx + 1} generated an exception: {exc}"
                                )
                                logger.error(f"Traceback: {traceback.format_exc()}")

                # Sort results by segment index to maintain order
                if self.segment_results:
                    self.segment_results.sort(key=lambda x: x["segment_idx"])

            else:
                # Sequential processing
                if use_parallel:
                    logger.info(
                        "Parallel processing requested but using sequential (single segment or disabled)"
                    )

                self.segment_results = []
                for i, (start, end) in enumerate(self.processing_segments):
                    segment_data = self.field_maps_windowed[start:end]
                    segment_time = self.time_windowed[start:end]

                    result = self._process_single_segment(segment_data, segment_time, i)
                    _handle_segment_result(result, i, self.segment_results)

            # Check if we have any successful results
            if not self.segment_results:
                logger.warning("No segments were successfully processed!")
                return

            # Combine results from all segments for logging
            all_losses = [result["best_loss"] for result in self.segment_results]
            all_i = [result["best_i"] for result in self.segment_results]
            
            all_r_hat_f = [
                list(np.round(result["r_hat"][:, 0].mean(axis=0) * 100, 4))
                for result in self.segment_results
            ]
            all_r_hat_m = [
                list(np.round(result["r_hat"][:, 1].mean(axis=0) * 100, 4))
                for result in self.segment_results
            ]

            self.log_dict["best_loss"] = all_losses
            self.log_dict["best_i"] = all_i
            self.log_dict["r_hat_f"] = all_r_hat_f
            self.log_dict["r_hat_m"] = all_r_hat_m

            logger.info(f"\nProcessed {len(self.segment_results)} segments")
            for i, result in enumerate(self.segment_results):
                r_hat = result["r_hat"]
                # Use global device for reporting
                logger.info(
                    f"Segment {i+1} - Loss: {result['best_loss']:.6f} @ {result['best_i']} it, "
                    f"Device: {self.device}, "
                    f"Dipole positions (cm): {np.array2string(r_hat.mean(axis=0)*100, formatter={'float_kind':lambda x: '%.2f' % x})}"
                )

            return

        # Original behavior for non-segmented processing
        self._log_step_header("Solving")

        # Solve
        self.m_hat, self.r_hat = self.mdl.solve(
            self.field_true,
            self.initial_parameters,
            lr=self.config["solver"]["lr"],
            loss_params=self.config["solver"].get("loss_params", None),
            plot=self.save_plots,
            plt_param=False,
            savename=(
                os.path.join(self.output_dir, "2_optimizer_loss.pdf")
                if self.save_plots
                else None
            ),
        )

        # Save the results in the log dictionary
        self.log_dict["best_loss"] = self.mdl.best_loss.tolist()
        self.log_dict["best_i"] = self.mdl.best_i.tolist() if isinstance(self.mdl.best_i, np.ndarray) else self.mdl.best_i
        self.log_dict["r_hat_f"] = list(
            np.round(self.r_hat[:, 0].mean(axis=0) * 100, 4)
        )
        self.log_dict["r_hat_m"] = list(
            np.round(self.r_hat[:, 1].mean(axis=0) * 100, 4)
        )
        logger.info(
            f"Dipole Positions (cm): {np.array2string(self.r_hat.mean(axis=0)*100, formatter={'float_kind':lambda x: '%.2f' % x})}"
        )

        # Apply wavelet denoising
        if self.config["post_processing"].get("wavelet_denoise", False):
            logger.info("\nWavelet Denoising")
        self.m_hat = _apply_wavelet_denoising(self.m_hat, self.config)
 
    def _combine_segment_results(self, segment_data_dict, i, all_beats_data):
        """Helper method to collect beats data from all segments and add artifacts in between."""
        segment_start_sample = self.processing_segments[i][0]
      
    
        # Initialize or extend the all_beats_data dictionary for fetal and maternal data
        for key in ["fetal", "maternal"]:
            # Initialize dict if empty
            if not all_beats_data or key not in all_beats_data:
                all_beats_data[key] = {"dipole_moments": [], "components": [], "peaks": [], "hr": [], "outlier": [], "position": [], "artifacts":[]}
            
            # Add artifacts between clean segments
            if segment_start_sample > 0:
                if i > 0:
                    artifact_length = self.processing_segments[i][0] - self.processing_segments[i - 1][1]#TODO
                else:
                    artifact_length = segment_start_sample
                all_beats_data[key]["hr"].extend([np.nan] * artifact_length)
                all_beats_data[key]["position"].extend([[np.nan, np.nan, np.nan]] * artifact_length)
                all_beats_data[key]["artifacts"].extend([True] * artifact_length)
                all_beats_data[key]["dipole_moments"].extend([[np.nan, np.nan, np.nan]] * artifact_length)
                all_beats_data[key]["components"].extend([np.nan] * artifact_length)

            # Extend the beats data with the current segment data
            logger.debug(
                f"{key} segment {i}: hr={len(segment_data_dict[key]['hr'])}, "
                f"outlier={len(segment_data_dict[key]['outlier'])}, "
                f"peaks={len(segment_data_dict[key]['peaks'])}, "
                f"segment_length={self.processing_segments[i][1] - self.processing_segments[i][0]}, "
                f"components={len(segment_data_dict[key]['components'])}, "
                f"dipole_moments={len(segment_data_dict[key]['dipole_moments'])}"
            )
            all_beats_data[key]["peaks"].extend(segment_data_dict[key]["peaks"] + segment_start_sample)
            all_beats_data[key]["hr"].extend(segment_data_dict[key]["hr"])
            all_beats_data[key]["outlier"].extend(segment_data_dict[key]["outlier"])
            all_beats_data[key]["position"].extend(segment_data_dict[key]["position"])
            all_beats_data[key]["artifacts"].extend([False] * (self.processing_segments[i][1] - self.processing_segments[i][0]))
            all_beats_data[key]["dipole_moments"].extend(segment_data_dict[key]["dipole_moments"])
            all_beats_data[key]["components"].extend(segment_data_dict[key]["components"])


            # If last segment, add artifacts until the end of the segment
            if i == len(self.processing_segments) - 1:
                segment_end_sample = self.processing_segments[i][1]
                if segment_end_sample < len(self.field_maps_dec):
                    artifact_length = len(self.field_maps_dec) - segment_end_sample
                    all_beats_data[key]["hr"].extend([np.nan] * artifact_length)
                    all_beats_data[key]["position"].extend([[np.nan, np.nan, np.nan]] * artifact_length)
                    all_beats_data[key]["artifacts"].extend([True] * artifact_length)
                    all_beats_data[key]["dipole_moments"].extend([[np.nan, np.nan, np.nan]] * artifact_length)
                    all_beats_data[key]["components"].extend([np.nan] * artifact_length)
   
    def _post_process(self):
        """Post-process the dipole moments to extract and average heartbeats, applying ICA post-decomposition if specified."""
        self._log_step_header("Post-Processing")

        if hasattr(self, "segment_results"):
            # For segmented processing, process each segment individually and collect all beats
            logger.info("Processing dipole moments from individual segments")

            # Check if we have any successful segment results
            if not self.segment_results:
                logger.warning(
                    "No successful segment results found. Skipping post-processing."
                )
                self.data_dict = {}
                self.heartbeats_dict = {}
                return

            # Process each segment individually and collect all segment data
            segment_data_list = []
            # Use dictionary to collect data for both fetal and maternal
            all_beats_data = {}
            for i, result in enumerate(self.segment_results):
                if result is not None and "m_hat" in result:
                    logger.info(f"Post-processing segment {i+1}")
                    
                    # Process individual segments - ICA & LMS
                    segment_data_dict = _process_dipole_data(
                        result["m_hat"],
                        result["r_hat"], 
                        self.fs_,
                        self.config,
                        self.log_dict,
                        False,
                        artifacts_mask=None
                    )
                    
                    # Extract ALL beats from this segment WITHOUT any selection

                    self._combine_segment_results(
                        segment_data_dict, i, all_beats_data
                    )
                    
                    segment_data_list.append(segment_data_dict)
            self.data_dict = all_beats_data

            # Check if we have any valid data after filtering
            if not segment_data_list:
                logger.warning(
                    "No valid dipole moment data found in segments. Skipping post-processing."
                )
                self.data_dict = {}
                self.heartbeats_dict = {}
                return

            # Combine the processed data from all segments for global analysis
            logger.info(f"Collected {len(all_beats_data['fetal']['peaks'])} fetal beats and {len(all_beats_data['maternal']['peaks'])} maternal beats from all segments")
            

            self.heartbeats_dict = _segment_and_average_beats(self.data_dict, self.fs_, self.config, self.log_dict)


            # Log heartbeat findings for segmented processing
            for component_name in self.heartbeats_dict.keys():
                if self.log_dict and f"selected_beats_{component_name}" in self.log_dict and f"selected_hr_{component_name}" in self.log_dict:
                    logger.info(f"Selected {component_name} {self.log_dict[f'selected_beats_{component_name}']:.2f} heartbeats with mean HR {self.log_dict[f'selected_hr_{component_name}']:.2f} bpm")

            # Compute GOF metrics per segment; store as lists (one entry per segment) in log_dict
            gof_lists = {}
            for result in self.segment_results:
                if result is not None and "m_hat" in result:
                    seg_idx = result["segment_idx"]
                    start, end = self.processing_segments[seg_idx]
                    field_seg = self.field_maps_windowed[start:end]
                    seg_metrics = self._compute_gof_metrics_for(
                        field_seg, result["m_hat"], result["r_hat"]
                    )
                    for k, v in seg_metrics.items():
                        gof_lists.setdefault(k, []).append(v)
            if gof_lists:
                self.log_dict.update(gof_lists)
                logger.info(
                    f"GOF segments ({len(self.segment_results)}): "
                    f"R²={gof_lists.get('recon_r_squared', [])}, "
                    f"fit_within_noise={gof_lists.get('fit_within_noise', [])}"
                )

            # Additional saving for segmented processing
            if self.save_data:
                # # Save individual segment results (simplified for pickling)
                # if hasattr(self, "segment_results") and self.segment_results:
                #     simplified_segment_results = []
                #     for result in self.segment_results:
                #         if result is not None:
                #             simplified_result = {
                #                 "m_hat": result["m_hat"],
                #                 "r_hat": result["r_hat"],
                #                 "best_loss": result["best_loss"],
                #                 "segment_idx": result["segment_idx"],
                #                 "time_length": (
                #                     len(result["time"]) if "time" in result else 0
                #                 ),
                #             }
                #             simplified_segment_results.append(simplified_result)

                # Save both data_dict and heartbeats_dict together in one file
                save_dict = {
                    "data_dict": self.data_dict,
                    "heartbeats_dict": self.heartbeats_dict,
                    #"field_maps": self.field_maps,
                }
                # Add ground truth if using synthetic data
                if hasattr(self, 'ground_truth') and self.ground_truth:
                    save_dict["ground_truth"] = self.ground_truth
                    logger.info("Including ground truth in pipeline_results.pkl (segmented)")

                with open(os.path.join(self.output_dir, "pipeline_results.pkl"), "wb") as f:
                    pickle.dump(save_dict, f)

            # # Add selected peak times to data_dict for proper report generation
            # for key in ["fetal", "maternal"]:
            #     if (key in self.heartbeats_dict and "M_x" in self.heartbeats_dict[key] and 
            #         "selected_peak_times" in self.heartbeats_dict[key]["M_x"] and 
            #         key in self.data_dict):
            #         self.data_dict[key]["selected_peak_times"] = self.heartbeats_dict[key]["M_x"]["selected_peak_times"]

            # Plot the results
            _plot_averaged_beats_if_enabled(
                self.heartbeats_dict, self.save_plots, self.output_dir
            )

            return

        # Original behavior for non-segmented processing
        self._execute_common_post_processing()

        # post_processing.average_reconstructed_fields(self.data_dict, self.mdl, self.fs_, savename=os.path.join(self.output_dir, "4_averaged_reconstructed_fields.pdf") if self.save_output else None)

    def _save_config(self):
        """Save the configuration settings to a JSON file in the output directory."""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        logger.info("\n============================")
        logger.info("Saving Config")
        logger.info("============================")

        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4, cls=NumpyEncoder)

    def _log_record(self):
        """Log the trial information to a CSV file in the output directory."""
        # Prepare log entry with trial information
        if self.log_note:
            self.log_note += self.log_dict.pop("note", "")

        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient": self.config["data"]["patient"],
            "series": self.config["data"]["series"],
            "signal_group": self.config["data"]["sig_group_names"],
            "sampfrom": self.config["data"]["sampfrom"],
            "sampto": self.config["data"]["sampto"],
            "signal_power": self.sig_power,
            "noise_power": self.noise_power,
            **{
                f"{k}": v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.log_dict.items()
            },
            "outlier_signal": (
                [
                    f"{sensor}"
                    for i, sensor in enumerate(self.sensor_dict.keys())
                    if (self.axis_mask[i] == 0).all()
                ]
                if self.config["data"].get("noise_group_names", None) is not None
                else []
            ),
        }
        log_entry["note"] = self.log_note if self.log_note else ""

        # Define CSV path
        csv_path = os.path.join(
            self.config.get("output_dir", "./"), "processed_records.csv"
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Read existing fieldnames if file exists, else use current log_entry keys
        file_exists = os.path.isfile(csv_path)
        if file_exists:
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = (
                    reader.fieldnames
                    if reader.fieldnames is not None
                    else list(log_entry.keys())
                )
        else:
            fieldnames = list(log_entry.keys())

        # Add any new keys from log_entry not in fieldnames
        for k in log_entry.keys():
            if k not in fieldnames:
                fieldnames.append(k)

        # Write to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # Fill missing keys with blank
            row = {k: log_entry.get(k, "") for k in fieldnames}
            writer.writerow(row)

        logger.info(f"\nLogged to: {csv_path}")

    def _save_reports(self):
        if not self.save_reports:
            return

        logger.info("\n============================")
        logger.info("Saving Reports")
        logger.info("============================")

        # Only generate reports if we have valid time and data
        if self.data_dict:
            if "hr" in self.save_reports:
                report_hr(
                    self.data_dict,
                    self.heartbeats_dict,
                    self.time_windowed,
                    self.output_dir,
                    segment_duration=180,
                )

            if "m_hat" in self.save_reports:
                report_m_hat_segments(
                    self.data_dict, self.time_windowed, self.output_dir
                )
            
            if "r_hat" in self.save_reports:
                report_r_hat_segments(
                    self.data_dict, self.time_windowed, self.output_dir
                )
            
            if "ica" in self.save_reports:
                report_ica_components(
                    self.data_dict, self.time_windowed, self.output_dir
            )
        else:
            logger.info("Skipping detailed reports - insufficient data for segmented processing")

    def load_measurements_from_labels(
        self, labels_path, measurement_config_date=None, sorted=True
    ):
        """
        Load measurement metadata from a CSV file and assign signal and noise group names.
        If no empty recordings are found for a patient or on the same date, noise columns are set to None.

        Args:
            labels_path (str): Path to the CSV file with columns: 'labels', 'groups', 'patient', 'date', 'series'.
            measurement_config_date (str, optional): Date to filter measurements. Only measurements after this date are included.
                If None, all measurements are included (default: None).
            sorted (bool): If True, sort the DataFrame by 'patient', 'series', and 'date' (default: True).

        Returns:
            pd.DataFrame: DataFrame with added columns:
            - 'sig_group_names': Signal group names for each patient.
            - 'noise_group_names': Noise (empty) group names, from the same or another patient on the same date.
            - 'noise_patient': Patient ID for the noise group (if not the current patient).
            - 'noise_series': Series ID for the noise group (if not the current patient).

        """

        df = pd.read_csv(labels_path)

        if measurement_config_date is not None:
            # Filter the DataFrame to only include rows with dates later than the specified date
            df = df[df["date"] >= measurement_config_date]

        df["sig_group_names"] = pd.Series(dtype=str)
        df["noise_group_names"] = pd.Series(dtype=str)
        df["noise_patient"] = pd.Series(dtype=str)
        df["noise_series"] = pd.Series(dtype=str)
        df["note"] = pd.Series(dtype=str)

        for i, row in df.iterrows():
            patient_labels = np.array(ast.literal_eval(row['labels']), dtype=int)

            df.at[i, "sig_group_names"] = np.array(ast.literal_eval(row["groups"]))[
                patient_labels  == 1
            ].tolist()
            # Check if the patient has empty recordings
            if (patient_labels == 0).any():
                df.at[i, "noise_group_names"] = ast.literal_eval(row["groups"])[
                    np.argmax(patient_labels == 0)
                ]
                df.at[i, "noise_patient"] = None
                df.at[i, "noise_series"] = None
                df.at[i, "note"] = f""
            else:
                # For this patient and session no empty recordings were found
                # Searching for empty recordings from the closest measurement date that includes a noise / empty recording
                # Remove leading 'D' from date strings before parsing
                df_dates = pd.to_datetime(df["date"].str.replace(r'^D', '', regex=True))
                row_date = pd.to_datetime(str(row["date"]).replace('D', '', 1))
                date_diffs = (df_dates - row_date).abs()
                # Exclude current row
                date_diffs[i] = pd.Timedelta(days=99999)
                # Only consider rows with at least one empty recording
                has_empty = df["labels"].apply(lambda x: any(np.array(ast.literal_eval(x), dtype=int) == 0))
                valid_indices = np.where(has_empty)[0]
                if len(valid_indices) > 0:
                    # Find the closest date among valid indices
                    closest_idx = valid_indices[np.argmin(date_diffs.iloc[valid_indices])]
                    closest_date = df.loc[closest_idx, "date"]
                    others = df[(df["date"] == closest_date) & has_empty]
                else:
                    others = pd.DataFrame()  # No valid empty recordings found

                # Check if there are other patients with the same date
                other_row = None
                if len(others) > 0:
                    for _, other_row_ in others.iterrows():
                        other_patient_labels = np.array(
                            ast.literal_eval(other_row_["labels"]), dtype=int
                        )
                        if (other_patient_labels == 0).any():
                            other_row = other_row_
                            break

                if other_row is not None:
                    logger.info(
                        f"Using empty recordings from patient {other_row['patient']} {other_row['series']} for {row['patient']} {row['series']}."
                    )
                    df.at[i, "noise_group_names"] = ast.literal_eval(
                        other_row["groups"]
                    )[np.argmax(other_patient_labels == 0)]
                    df.at[i, "noise_patient"] = other_row["patient"]
                    df.at[i, "noise_series"] = other_row["series"]
                    df.at[i, "note"] = (
                        f"Using empty device measurement from {other_row['patient']} {other_row['series']} for {row['patient']} {row['series']}"
                    )
                else:
                    logger.warning(
                        f"No empty recordings found for patient {row['patient']} on date {row['date']}."
                    )
                    df.at[i, "noise_group_names"] = None
                    df.at[i, "noise_patient"] = None
                    df.at[i, "noise_series"] = None
                    df.at[i, "note"] = "No empty device measurement"

        if sorted:
            df.sort_values(by=["patient", "series", "date"], inplace=True)
        return df

    def run(
        self,
        config,
        system_config,
        measurement_config,
        mask=None,
        log_note="",
        log_dict={},
        verbose=False,
        data_loader=None,
    ):
        """
        Run the complete pipeline from data loading to post-processing.

        Args:
            config: Pipeline configuration (dict or PipelineConfig dataclass)
            system_config: System configuration for measurement device
            measurement_config: Measurement configuration
            mask: Optional mask to exclude specific sensor axes
            log_note: Optional note to include in the log record
            log_dict: Optional dictionary for logging additional information
            verbose: If True, enables verbose logging
            data_loader: Optional synthetic data loader (for evaluation)

        Note:
            If no noise data is provided in the configuration, the pipeline will not execute and will return early.
            The config parameter accepts both legacy dict format and new PipelineConfig dataclass.
            If data_loader is provided, it will be used instead of loading from TDMS files.
        """

        if verbose:
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.systemconfig = system_config
        self.measurementconfig = measurement_config

        # Convert PipelineConfig (dataclass) to dict if needed for backward compatibility
        if hasattr(config, "to_dict"):
            self.config = config.to_dict()
            logger.debug("Converted PipelineConfig via to_dict() to dict format")
        elif dataclasses.is_dataclass(config):
            # Convert dataclass to plain dict
            self.config = dataclasses.asdict(config)
            logger.debug("Converted PipelineConfig dataclass to dict format via dataclasses.asdict")
        else:
            self.config = config

        # Use the dict-like config from here on
        # Handle both "device" (single) and "devices" (list for batch processing)
        if "device" in self.config:
            self.device = self.config["device"]
        elif "devices" in self.config and self.config["devices"]:
            self.device = self.config["devices"][0]  # Use first device
            logger.info(f"Using first device from 'devices' list: {self.device}")
        else:
            self.device = "cpu"
        
        # CRITICAL: Set device in config dict so workers can access it
        self.config["device"] = self.device
        
        self._step_num = 1
        self.save_reports = self.config.get("save_reports", False)
        self.save_plots = self.config.get("save_plots", False)
        self.save_data = self.config.get("save_data", False)
        self.output_dir = self.config.get("output_dir", "output")
        self.mask = mask
        self.log_note = log_note
        self.log_dict = log_dict
        self.data_loader = data_loader  # Store synthetic data loader if provided
        self.ground_truth = None  # Will be populated if using synthetic data

        self.sig_data_dict_raw = None
        self.noise_data_dict_raw = None
        self.time = None
        self.fs = None
        self.sig_power = None
        self.noise_power = None  #
        self.noise_whitened = None
        self.sensor_dict = {}



        logger.info("\n====================")
        logger.info("Pipeline Execution")
        logger.info("====================")


        # Set CUDA device early to prevent cuda:0 allocation - minimal approach
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.debug("Cleaned CUDA cache before pipeline start")

        # Initialize output directory
        logger.info(f"Output Directory: {self.output_dir}")
        self._initialize_output_directory()

        # If no noise data is provided, skip the pipeline execution
        if self.config["data"].get("noise_group_names", None) is None:
            self._log_record()
            self._save_config()
            logger.info("No noise data provided, skipping pipeline execution.")
            return

        if self.config.get("catch_errors", True):
            try:
                self._pipeline_steps()
            except Exception as e:
                self.log_note += f"Error during pipeline execution: {str(e)} "
                logger.error(f"Exception during pipeline execution: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            self._pipeline_steps()

        # Save the log record and configuration
        self._log_record()
        self._save_config()
        plt.close("all")  # Close all plots to free memory

    def _pipeline_steps(self):
        """The main pipeline steps."""
        self._load_data()
        self._preprocess_data()
        self._decimate_data()
        self._select_time_window()
        self._initialize_solver()
        self._solve()
        self._post_process()
        self._save_reports()
        logger.info("\nPipeline completed successfully.")

# Example usage
if __name__ == "__main__":
    from fmcg.configs.measurement_config_2024_07_05 import (
        MeasurementConfig as MeasurementConfig_2024_07_05,
        SystemConfig as SystemConfig_2024_07_05,
    )

    example_config = {
        "catch_errors": True,
        "device": "cuda:3",
        "save_reports": True,
        "save_plots": True,
        "save_data": True,
        "output_dir": "/jonas/fMCG/output",
        "data": {
            "ds_path": "/path/to/your/dataset",  # Update this path
            "patient": "P051",
            "series": "S02",
            "sig_group_names": "R003",  # 2. ####################
            "noise_group_names": "R004",  # 1. ####################
            "noise_patient": None,
            "noise_series": None,
            "sampfrom": 0,
            "sampto": None,
        },
        "preprocessing": {
            "whitening": "ZCA",
            "rescale_whitening": True,
            "bandpass_low": 2,
            "bandpass_high": 40,
            "bandpass_order": 4,
            "signal_threshold": 2,
            "noise_threshold": 0.1,
            "downsample_factor": 2,
            "process_segments": False,  # Set to True to enable segmented processing
            "min_segment_length": 30,  # Minimum segment length in seconds for segmented processing
        },
        "solver": {
            "num_dipoles": 2,
            "optimizer": "adam",
            "m_scaling": [1e1, 1e1],
            "parallel_processing": True,  # Set to True to enable parallel segment processing
            "max_workers": 4,  # Number of parallel workers for segment processing
            # Note: GPU parallelization is limited by CUDA's threading model and may not provide
            # significant speedup on a single GPU. CPU parallelization uses multiprocessing for better performance.
            # For optimal GPU performance, consider using sequential processing or multiple GPU devices.
            "lr": {
                "start": 1e-1,
                "pos_start": 1e-1,  # 3. #########################
                "end": 1e-2,
                "niter": 300,
                "warmup": 50,
            },
            "initializer": {
                "method": "pseudo_inv_regularized",
                "rcond": 1e-3,
                "determine_scaling": True,  # 4. ###############
                "update_m_scaling": True,
                "field_scaling": 1e0,
            },
            # "m_basis": {"basis": "exponential", "base": "10"},
            "r_basis": "sigmoid_time_constant",
            "r_bnds": np.array(
                [
                    [(-0.3, 0.3), (-0.3, -0.005), (-0.3, 0.3)],
                    [(-0.3, 0.3), (-0.3, -0.005), (0.1, 0.6)],
                ]
            ),
            "r_init": np.array([[[0, -0.02, 0.0], [0, -0.05, 0.2]]]),
        },
        "post_processing": {
            "method": "subsequent_LMS",
            "mu": 0.001,
            "wavelet_denoise": False,  # 5. #######################
            "wavelet_denoise_threshold": 100,
            "segment_length": None,
            "ica_components": 3,
            "n_trials": 5,
            "nlms": False,
            "ratio_pre": 0.5,  # Ratio of pre-peak samples for beat segmentation
            "interval": 2.5,   # Total interval length for beat segmentation in seconds
            "averaging": {
                "method": "selective_averaging",  # "remove_outlier"
                "tol": 3,
                "outlier_kwargs": {"threshold": 10, "win": 4},
            },
        },
    }
    pipeline = Pipeline()
    pipeline.run(
        example_config,
        SystemConfig_2024_07_05,
        MeasurementConfig_2024_07_05
    )

