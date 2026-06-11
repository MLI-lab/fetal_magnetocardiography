"""
Shared fMCG preprocessing.

Used by MultiApproachPipeline, Pipeline, ICAComponentLabeler, and notebooks
to ensure identical preprocessing across all callers.

Public API:
    load_and_preprocess()  — load data from disk and preprocess in one call
    preprocess()           — preprocess already-loaded sensor dicts
    PreprocessedData       — output dataclass (supports tuple unpacking)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import scipy.signal

from fmcg.signal.artifact_removal import estimate_noise_mask, remove_outlier_dict
from fmcg.signal.filtering import filter_sensor_dict
from fmcg.signal.whitening import apply_whitening
from fmcg.pipeline._pipeline_utils import _find_continuous_segments

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """
    Output of the shared preprocessing step.

    Supports tuple unpacking in a fixed, documented order:
        field_data, noise_maps, axis_mask, whitening_matrix, fs, time,
        r_sensors, artifacts_mask, sensor_names = preprocessed

    Attributes
    ----------
    field_data : np.ndarray
        (T, S, 3) full windowed array after preprocessing. Artifact regions
        retain their filtered values; use artifacts_mask to exclude them.
    noise_maps : np.ndarray or None
        (T_noise, S, 3) filtered noise array in physical units (pre-whitening).
        Used for GOF discrepancy metrics and noise-level diagnostics.
    axis_mask : np.ndarray
        (S, 3) int mask — 1 = valid channel/axis, 0 = removed as outlier.
    whitening_matrix : np.ndarray or None
        (n_valid_ch, n_valid_ch) whitening matrix W; None if whitening disabled.
    fs : int
        Decimated sampling rate [Hz].
    time : np.ndarray
        (T,) time vector [s].
    r_sensors : np.ndarray or None
        (S, 3) sensor positions [m]; None if not available from data loader.
    artifacts_mask : np.ndarray
        (T,) bool — True = artifact sample.
    sensor_names : list or None
        Ordered list of sensor names matching axis 1 of field_data.
    """
    field_data: np.ndarray
    noise_maps: Optional[np.ndarray]
    axis_mask: np.ndarray
    whitening_matrix: Optional[np.ndarray]
    fs: int
    time: np.ndarray
    r_sensors: Optional[np.ndarray]
    artifacts_mask: np.ndarray
    sensor_names: Optional[List[str]]

    def __iter__(self):
        """Yield fields in documented order for tuple unpacking."""
        yield self.field_data
        yield self.noise_maps
        yield self.axis_mask
        yield self.whitening_matrix
        yield self.fs
        yield self.time
        yield self.r_sensors
        yield self.artifacts_mask
        yield self.sensor_names


def load_and_preprocess(
    ds_path: str,
    patient: str,
    series: str,
    sig_group_names,
    noise_group_names,
    *,
    noise_patient: Optional[str] = None,
    noise_series: Optional[str] = None,
    bandpass_low: float = 1.0,
    bandpass_high: float = 40.0,
    bandpass_order: int = 4,
    signal_threshold: float = 1.0,
    noise_threshold: float = 0.1,
    whitening: Optional[str] = "PCA",
    rescale_whitening: bool = False,
    downsample_factor: int = 1,
    detect_artifacts=False,
    sampfrom: float = 0,
    sampto: Optional[float] = None,
    channel_mask=None,
) -> PreprocessedData:
    """
    Load fMCG data from disk and run the full preprocessing pipeline.

    Combines :func:`~fmcg.utils.data.load_structured_patient_data_and_noise`
    with :func:`preprocess` into a single call for use in notebooks and scripts.

    Parameters
    ----------
    ds_path : str
        Path to the dataset directory.
    patient, series : str
        Patient and series identifiers.
    sig_group_names, noise_group_names : str or list
        TDMS group name(s) for signal and noise recordings.
    noise_patient, noise_series : str, optional
        Override patient/series for the noise recording.
    bandpass_low, bandpass_high : float
        Bandpass filter cutoffs [Hz]. Defaults: 1 and 40 Hz.
    bandpass_order : int
        Butterworth filter order. Default: 4.
    signal_threshold, noise_threshold : float
        Physical range thresholds for outlier channel removal [pT].
    whitening : str or None
        Whitening method: "PCA", "ZCA", or None to disable.
    rescale_whitening : bool
        Rescale whitened output to original energy scale.
    downsample_factor : int
        Integer decimation factor (e.g. 4 → fs/4). Default: 1 (no decimation).
        Use ``fs // fs_target`` to convert from a target frequency.
    detect_artifacts : bool or dict
        Run artifact detection. Pass a dict to forward kwargs to
        :func:`~fmcg.signal.artifact_removal.estimate_noise_mask`.
    sampfrom, sampto : float
        Time window to extract [s]. sampto=None uses the full signal.
    channel_mask : array-like, optional
        Additional (S, 3) mask ANDed with the outlier-derived axis_mask.

    Returns
    -------
    PreprocessedData
        Supports tuple unpacking::

            field_data, noise_maps, axis_mask, W, fs, time, r_sensors,
            artifacts_mask, sensor_names = load_and_preprocess(...)

    Examples
    --------
    >>> field_data, noise_maps, axis_mask, W, fs, time, r_sensors, _, _ = (
    ...     load_and_preprocess(ds_path="/data", patient="P097", series="S07",
    ...                         sig_group_names="R001", noise_group_names="R004",
    ...                         whitening="ZCA", downsample_factor=5))
    """
    from fmcg.utils import data as data_utils

    sig_dict, time, fs, noise_dict, configs = (
        data_utils.load_structured_patient_data_and_noise(
            ds_path=ds_path,
            patient=patient,
            series=series,
            sig_group_names=sig_group_names,
            noise_group_names=noise_group_names,
            noise_patient=noise_patient,
            noise_series=noise_series,
            files="all",
            return_configs=True,
        )
    )
    r_sensors = configs.get("r_sensors")

    params = dict(
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        bandpass_order=bandpass_order,
        signal_threshold=signal_threshold,
        noise_threshold=noise_threshold,
        whitening=whitening,
        rescale_whitening=rescale_whitening,
        downsample_factor=downsample_factor,
        detect_artifacts=detect_artifacts,
        sampfrom=sampfrom,
        sampto=sampto,
        channel_mask=channel_mask,
    )
    return preprocess(sig_dict, noise_dict, fs, params, time=time, r_sensors=r_sensors)


def preprocess(
    sig_dict: dict,
    noise_dict: dict,
    fs: float,
    params: dict,
    time: Optional[np.ndarray] = None,
    r_sensors: Optional[np.ndarray] = None,
) -> PreprocessedData:
    """
    Run the full preprocessing pipeline on already-loaded sensor dicts.

    Data-loader agnostic: can be called from MultiApproachPipeline,
    Pipeline, ICAComponentLabeler, or synthetic loaders.

    Parameters
    ----------
    sig_dict : dict
        {sensor_name: (T, 3) array} raw signal per sensor.
    noise_dict : dict
        {sensor_name: (T, 3) array} raw noise per sensor.
    fs : float
        Sampling rate of the input data [Hz].
    params : dict
        Flat preprocessing config. Keys (all optional unless noted):
          bandpass_low* (float), bandpass_high* (float), bandpass_order (int, default 4)
          signal_threshold (float, default 10.0), noise_threshold (float, default 10.0)
          channel_mask (array-like, optional additional mask)
          detect_artifacts (True/dict of kwargs forwarded to estimate_noise_mask,
                           or False to disable, default True)
          whitening (str or None), rescale_whitening (bool)
          downsample_factor (int, default 1 = no decimation)
          sampfrom (float [s], default 0), sampto (float [s], default end)
    time : np.ndarray, optional
        Time vector (T,). Constructed from 0..T/fs if not provided.
    r_sensors : np.ndarray, optional
        Sensor positions (S, 3) [m].

    Returns
    -------
    PreprocessedData

    Notes
    -----
    Pipeline callers pass:
        preprocess(sig_dict, noise_dict, fs,
                   {**config["preprocessing"],
                    "sampfrom": config["data"].get("sampfrom", 0),
                    "sampto": config["data"].get("sampto")})

    ICAComponentLabeler callers pass params directly (same flat key names).
    """
    sensor_names = list(sig_dict.keys())

    if time is None:
        n_samples = len(next(iter(sig_dict.values())))
        time = np.arange(n_samples) / fs

    # --- Bandpass filter ---
    bp_low = params["bandpass_low"]
    bp_high = params["bandpass_high"]
    bp_order = params.get("bandpass_order", 4)
    logger.info(f"  Bandpass: {bp_low}-{bp_high} Hz (order {bp_order})")

    sig_dict = filter_sensor_dict(sig_dict, fs, low=bp_low, high=bp_high, order=bp_order)
    noise_dict = filter_sensor_dict(noise_dict, fs, low=bp_low, high=bp_high, order=bp_order)

    # --- Outlier removal ---
    if params.get("skip_outlier_removal", False):
        n_sensors = len(sig_dict)
        n_axes = next(iter(sig_dict.values())).shape[1]
        axis_mask = np.ones((n_sensors, n_axes), dtype=bool)
    else:
        sig_dict, axis_mask_sig = remove_outlier_dict(
            sig_dict,
            **{"physical_range_threshold": params["signal_threshold"]} if "signal_threshold" in params else {},
        )
        noise_dict, axis_mask_noise = remove_outlier_dict(
            noise_dict,
            **{"physical_range_threshold": params["noise_threshold"]} if "noise_threshold" in params else {},
        )

        if axis_mask_sig.shape != axis_mask_noise.shape:
            raise ValueError(
                f"Axis mask shape mismatch: signal {axis_mask_sig.shape} vs noise {axis_mask_noise.shape}"
            )
        axis_mask = axis_mask_sig.astype(bool) & axis_mask_noise.astype(bool)

    if params.get("channel_mask") is not None:
        axis_mask = axis_mask & np.asarray(params["channel_mask"], dtype=bool)

    logger.info(f"  Valid channels: {int(np.sum(axis_mask))} / {axis_mask.size}")

    # --- Convert dicts to (T, S, 3) arrays ---
    field_maps = np.array(list(sig_dict.values())).transpose(1, 0, 2)
    noise_maps = np.array(list(noise_dict.values())).transpose(1, 0, 2)

    # --- Artifact (noise burst) detection ---
    detect_artifacts_cfg = params.get("detect_artifacts", {})
    if detect_artifacts_cfg is not False:
        field_norm = np.linalg.norm(field_maps[:, axis_mask == 1], axis=1)
        artifacts_mask = estimate_noise_mask(
            field_norm, fs=fs, **(detect_artifacts_cfg if isinstance(detect_artifacts_cfg, dict) else {})
        )
    else:
        artifacts_mask = np.zeros(field_maps.shape[0], dtype=bool)

    logger.info(f"  Artifacts: {np.mean(artifacts_mask) * 100:.1f}% of signal")

    # --- Whitening ---
    whitening_method = params.get("whitening")
    if whitening_method:
        logger.info(f"  Whitening: {whitening_method}")
        field_clean, W = apply_whitening(
            field_maps[~artifacts_mask],
            noise_maps,
            axis_mask,
            method=whitening_method,
            return_whitening_matrix=True,
            rescale=params.get("rescale_whitening"),
        )
        field_maps[~artifacts_mask] = field_clean
    else:
        W = None

    # --- Decimate ---
    q = params.get("downsample_factor", 1)
    if q > 1:
        fs_dec = int(fs / q)
        field_maps = scipy.signal.decimate(field_maps, q, axis=0).copy()
        noise_maps = scipy.signal.decimate(noise_maps, q, axis=0).copy()
        n_dec = field_maps.shape[0]
        artifacts_dec = np.zeros(n_dec, dtype=bool)
        for i in range(n_dec):
            artifacts_dec[i] = np.any(artifacts_mask[i * q: (i + 1) * q])
        time = np.linspace(time[0], time[-1], num=n_dec, endpoint=True)
        artifacts_mask = artifacts_dec
        logger.info(f"  Decimated: {n_dec} samples at {fs_dec} Hz")
    else:
        fs_dec = fs

    # --- Select time window ---
    sampfrom = params.get("sampfrom", 0) or 0
    sampto = params.get("sampto")
    t0 = int(sampfrom * fs_dec)
    t1 = int(sampto * fs_dec) if sampto is not None else len(field_maps)

    logger.info(
        f"  Window: {sampfrom:.1f}s - {sampto or round(time[-1], 1):.1f}s "
        f"({(t1 - t0) / fs_dec:.1f}s)"
    )

    return PreprocessedData(
        field_data=field_maps[t0:t1],
        noise_maps=noise_maps,
        axis_mask=axis_mask,
        whitening_matrix=W,
        fs=int(fs_dec),
        time=time[t0:t1],
        r_sensors=r_sensors,
        artifacts_mask=artifacts_mask[t0:t1],
        sensor_names=sensor_names,
    )
