import logging
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import binary_closing, binary_dilation
from scipy.stats import median_abs_deviation

from fmcg.signal.filtering import logger
from fmcg.utils.data import logger


def wavelet_artifact_removal(
    sig,
    fs,
    wavelet="haar",
    level=7,
    segment_length_seconds=5,
    plot=True,
    coeff_depth=None,
    n_sigma=3,
):
    """
    Remove artifacts from a signal using stationary wavelet transform (SWT) and segment-based thresholding.

    Parameters:
        sig (np.ndarray): Input 1D signal.
        fs (int): Sampling frequency.
        wavelet (str): Wavelet name for SWT.
        level (int): Number of decomposition levels.
        segment_length_seconds (int): Segment length for thresholding (in seconds).
        plot (bool): If True, plot the decomposition and results.
        coeff_depth (int or list, optional): Which coefficient(s) to use for thresholding.
            If None, use all coefficients (default behavior).
            If int, selects the last `coeff_depth` components.
        n_sigma (int, optional): Number of standard deviations for thresholding. Default is 3.

    Returns:
        cleaned_signal (np.ndarray): Signal with artifacts removed.
        artifact_signal (np.ndarray): Estimated artifact signal.
        artifact_mask (np.ndarray): Binary mask indicating artifact presence.
        thresholded_coeffs (list): List of thresholded coefficient arrays.
    """

    # Ensure signal length is a power of 2 by padding with zeros
    next_power_of_2 = int(2 ** np.ceil(np.log2(len(sig))))
    original_length = len(sig)
    sig = np.pad(sig, (0, next_power_of_2 - len(sig)), mode="constant")

    # SWT decomposition
    coeffs = pywt.swt(sig, wavelet, level=level, start_level=0)
    wavelet_coeffs = [c[1] for c in coeffs]
    approx_coeffs = coeffs[-1][0]
    all_coeffs = wavelet_coeffs + [approx_coeffs]

    # Determine which coefficients to use for thresholding
    if coeff_depth is None:
        coeff_indices = list(range(len(all_coeffs)))
    elif isinstance(coeff_depth, int):
        coeff_indices = list(range(0, coeff_depth))
    else:
        coeff_indices = list(coeff_depth)

    def calculate_thresholds(coefficient_sequence, segment_length_seconds=5, fs=1000):
        segment_samples = int(segment_length_seconds * fs)
        num_segments = len(coefficient_sequence) // segment_samples
        if num_segments == 0:
            return -np.inf, np.inf
        maxima, minima = [], []
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = (i + 1) * segment_samples
            segment = coefficient_sequence[start_idx:end_idx]
            if len(segment) > 0:
                maxima.append(np.max(segment))
                minima.append(np.min(segment))
        if len(coefficient_sequence) % segment_samples != 0:
            remaining = coefficient_sequence[num_segments * segment_samples :]
            if len(remaining) > 0:
                maxima.append(np.max(remaining))
                minima.append(np.min(remaining))
        if len(maxima) == 0 or len(minima) == 0:
            return -np.inf, np.inf
        mu_max = np.median(maxima)
        sigma_max = median_abs_deviation(maxima, scale="normal")
        mu_min = np.median(minima)
        sigma_min = median_abs_deviation(minima, scale="normal")
        c_max = mu_max + n_sigma * sigma_max
        c_min = mu_min - n_sigma * sigma_min
        upper_threshold = max([m for m in maxima if m <= c_max], default=np.inf)
        lower_threshold = min([m for m in minima if m >= c_min], default=-np.inf)
        return lower_threshold, upper_threshold

    thresholded_coeffs = []
    for idx, series in enumerate(all_coeffs):
        if idx in coeff_indices:
            lower_thresh, upper_thresh = calculate_thresholds(
                series, segment_length_seconds, fs
            )
            outlier_coeff = np.zeros_like(series)
            outlier_mask = (series > upper_thresh) | (series < lower_thresh)
            outlier_coeff[outlier_mask] = series[outlier_mask]
            thresholded_coeffs.append(outlier_coeff)
        else:
            thresholded_coeffs.append(np.zeros_like(series))

    # Prepare coefficients for SWT reconstruction
    outlier_coeffs_swt = []
    for i in range(level):
        if i < len(thresholded_coeffs) - 1:
            outlier_coeffs_swt.append(
                (np.zeros_like(thresholded_coeffs[i]), thresholded_coeffs[i])
            )
        else:
            outlier_coeffs_swt.append(
                (thresholded_coeffs[-1], np.zeros_like(thresholded_coeffs[-1]))
            )

    artifact_signal = pywt.iswt(outlier_coeffs_swt, wavelet)
    if len(artifact_signal) > len(sig):
        artifact_signal = artifact_signal[: len(sig)]
    elif len(artifact_signal) < len(sig):
        artifact_signal = np.pad(
            artifact_signal, (0, len(sig) - len(artifact_signal)), "constant"
        )
    cleaned_signal = sig - artifact_signal

    # remove padding
    #cleaned_signal = cleaned_signal[:original_length]

    if plot:
        time_axis = np.arange(len(sig)) / fs
        plt.figure(figsize=(12, 2 * len(all_coeffs)))
        for idx, series in enumerate(all_coeffs):
            lower_thresh, upper_thresh = (
                calculate_thresholds(series, segment_length_seconds, fs)
                if idx in coeff_indices
                else (-np.inf, np.inf)
            )
            plt.subplot(len(all_coeffs), 1, idx + 1)
            plt.plot(series, label=f"Coeff {idx+1}")
            if idx in coeff_indices:
                plt.axhline(
                    upper_thresh, color="red", linestyle="--", label="Upper Threshold"
                )
                plt.axhline(
                    lower_thresh, color="red", linestyle="--", label="Lower Threshold"
                )
            plt.title(f"Coefficient Series {idx+1}")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.legend(loc="upper right")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        axs[0].plot(time_axis, sig, label="Original Signal")
        axs[0].set_ylabel("Signal")
        axs[0].set_title("Original Signal")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(time_axis, artifact_signal, "tab:red", label="Estimated Artifacts")
        axs[1].set_ylabel("Signal")
        axs[1].set_title("Estimated Motion Artifacts")
        axs[1].legend()
        axs[1].grid(True)
        axs[2].plot(time_axis, cleaned_signal, "tab:green", label="Cleaned Signal")
        axs[2].set_ylabel("Signal")
        axs[2].set_title("Cleaned Signal (Original - Artifacts)")
        axs[2].legend()
        axs[2].grid(True)
        axs[3].plot(time_axis, sig, "tab:orange", label="Original Signal", alpha=0.7)
        axs[3].plot(
            time_axis, cleaned_signal, "tab:green", label="Cleaned Signal", alpha=0.8
        )
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Signal")
        axs[3].set_title("Comparison: Original vs Cleaned")
        axs[3].legend()
        axs[3].grid(True)
        plt.tight_layout()
        plt.show()

    artifact_binary = (artifact_signal != 0).astype(float)
    artifact_mask = artifact_binary

    return cleaned_signal, artifact_mask, artifact_signal, thresholded_coeffs


def detect_artifacts(
    signal_array,
    fs=1000,
    wavelet="haar",
    level=5,
    segment_length_seconds=1,
    struct_time=10,
    channel_agreement_percentage=0.5,
    coeff_depth=None,
    offset=None,
    n_sigma=3,
    plot=False,
):
    """
    Detect artifacts in a multi-channel signal using wavelet transform and segment-based thresholding.
    Parameters:
        signal_array (np.ndarray): 2D array of shape (n_samples, n_channels) containing the signal data.
        fs (int): Sampling frequency of the signal.
        wavelet (str): Wavelet name for SWT.
        level (int): Number of decomposition levels for SWT.
        segment_length_seconds (int): Length of segments for determining thresholds in seconds.
        struct_time (int): Time in seconds for binary closing structure.
        channel_agreement_percentage (float): Percentage of channels that must agree on artifact presence.
        coeff_depth (int or list, optional): Which coefficient(s) to use for thresholding. Default is None, meaning all coefficients are used.
        offset (int, optional): Time padding in seconds to extend the artifact mask on each side.
        n_sigma (int, optional): Number of standard deviations for thresholding. Default is 3.
        plot (bool, optional): If True, plot the decomposition and results. Default is False.

    Returns:
        artifact_mask (np.ndarray): Binary mask indicating artifact presence across channels.
    """

    artifact_masks = []
    for axis in range(signal_array.shape[-1]):
        signal = signal_array[:, axis]
        _, artifact_mask, _, _ = wavelet_artifact_removal(
            signal,
            fs=fs,
            wavelet=wavelet,
            level=level,
            segment_length_seconds=segment_length_seconds,
            plot=plot,
            coeff_depth=coeff_depth,
            n_sigma=n_sigma,
        )
        artifact_masks.append(artifact_mask)

    mean_artifact = np.array(artifact_masks).mean(axis=0)

    binary_mask = mean_artifact > channel_agreement_percentage
    structure = np.ones(int(struct_time * fs + 1))
    artifact_mask = binary_closing(binary_mask, structure=structure)

    # pad artifact mask to match the original signal length
    if len(artifact_mask) < signal_array.shape[0]:
        artifact_mask = np.pad(
            artifact_mask, (0, signal_array.shape[0] - len(artifact_mask)), "edge"
        )
    elif len(artifact_mask) > signal_array.shape[0]:
        artifact_mask = artifact_mask[: signal_array.shape[0]]

    if offset is not None:
        artifact_mask = binary_dilation(
            artifact_mask, structure=np.ones(int(offset * fs))
        )

    return artifact_mask


def remove_outlier_dict(
    sensor_dict, physical_range_threshold=0.1, threshold=0.5, verbose=False
):
    """
    Detect outlying channels in a dictionary of sensor data and replace them with NaN based on a specified percentile and threshold.

    This function identifies outlying channels with low or high variability by comparing the proportion of samples with differences
    below or above a global threshold. The global threshold is determined using the n-th percentile of the absolute differences between
    consecutive samples, aggregated by the median across all sensors. Channels with a physical range smaller than the specified threshold
    are not considered for outlier threshold computation.

    Parameters:
        sensor_dict (dict): Dictionary where keys are sensor names and values are 2D numpy arrays (n_samples, n_channels) representing
            sensor data over time for multiple channels.
        physical_range_threshold (float, optional): Threshold (in pT) for the minimum physical range of the sensor data. Default is 0.1 pT.
            Channels with a range smaller than this threshold are not considered for outlier threshold computation.
        threshold (float, optional): The threshold above which a channel is considered an outlier. Default is 0.5.
        verbose (bool, optional): If True, print warnings for channels with a physical range below the threshold.

    Returns:
        dict: The updated sensor dictionary with outlying channels replaced by NaN.
        numpy.ndarray: A boolean array indicating which channels are NOT outliers.
    """

    if verbose:
        logger.setLevel(logging.DEBUG)

    # 1: Per-channel physical ranges & mask
    physical_mask = {}
    physical_ranges = {}
    for k, v in sensor_dict.items():
        chan_ranges = np.nanpercentile(v, 95, axis=0) - np.nanpercentile(v, 5, axis=0)
        physical_ranges[k] = chan_ranges

        mask = chan_ranges > physical_range_threshold
        physical_mask[k] = mask

    # Step 2: Compute global thresholds from physically valid channels
    all_valid_diffs_low = []
    all_valid_diffs_up = []
    for k, v in sensor_dict.items():
        mask = physical_mask[k]
        if not np.any(mask):
            continue
        diffs = np.abs(np.diff(v[:, mask], axis=0))
        all_valid_diffs_low.append(np.nanpercentile(diffs, 5, axis=0))
        all_valid_diffs_up.append(np.nanpercentile(diffs, 95, axis=0))

    if not all_valid_diffs_low:
        logger.warning("No channels passed physical range threshold.")
        return sensor_dict, np.zeros(
            (len(sensor_dict), next(iter(sensor_dict.values())).shape[1]), dtype=bool
        )

    global_threshold_low = np.nanmedian(np.concatenate(all_valid_diffs_low))
    global_threshold_up = np.nanmedian(np.concatenate(all_valid_diffs_up))

    # Step 3: Identify outliers per channel
    results = {}
    for key, value in sensor_dict.items():
        diffs = np.abs(np.diff(value, axis=0))
        outlier_frac = (
            np.sum(diffs < global_threshold_low, axis=0)
            + np.sum(diffs > global_threshold_up, axis=0)
        ) / value.shape[0]

        all_nan = np.isnan(value).all(axis=0)
        outlier_frac[all_nan] = 1.0
        outlier_frac[~physical_mask[key]] = 1.0
        results[key] = outlier_frac.round(2)
        sensor_dict[key][:, (outlier_frac > threshold)] = np.nan

    # Step 4: Pretty print fixed-width table
    sensor_names = list(sensor_dict.keys())

    # Header row
    column_labels = [
        f"{s} ⚠️" if (results[s] > threshold).sum() > 1 else s for s in sensor_names
    ]
    column_labels = [f"{s:<6}" for s in column_labels]
    header = " " * 6 + " | ".join(column_labels)
    logger.info(header)
    logger.info("-" * len(header))

    # Row per channel
    channel_labels = ["X", "Y", "Z"]  # extend as needed
    max_channels = max(v.shape[1] for v in sensor_dict.values())
    for ch_idx in range(max_channels):
        row_label = (
            channel_labels[ch_idx] if ch_idx < len(channel_labels) else f"Ch{ch_idx}"
        )
        row_cells = []
        for s in sensor_names:
            if ch_idx >= physical_ranges[s].shape[0]:
                cell = ""
            else:
                rng = physical_ranges[s][ch_idx]
                frac = results[s][ch_idx]
                cell = f"{rng:6.2f}"
            row_cells.append(cell)
        logger.info(f"{row_label:<6}" + " | ".join(row_cells))

    for k in results:
        # Additionally warn about explicitly unphysical channels (range too small)
        unphys_idx = np.where(~physical_mask[k])[0]
        for idx in unphys_idx:
            if np.isnan(physical_ranges[k][idx]):
                continue
            logger.warning(
                f"WARNING (Physical Range Threshold): Sensor '{k}' channel {np.array(['X', 'Y', 'Z'])[idx]} have physical range {physical_ranges[k][idx]:.3f} "
                f"(threshold {physical_range_threshold}). Marked unphysical and ignored."
            )

    # Step 5: Apply masking
    for k, frac in results.items():
        mask = (
            (frac > threshold)
            & physical_mask[k]
            & ~np.isnan(sensor_dict[k]).all(axis=0)
        )  # dont list unphysical or nan channels
        if np.sum(mask) > 1:
            bad_channels = np.array(["X", "Y", "Z"][: sensor_dict[k].shape[1]])[mask]
            logger.warning(
                f"WARNING (Variability Outlier): {k} channel(s) {bad_channels} are outlying ({frac[mask]} %) and will be IGNORED."
            )
        # sensor_dict[k][:, mask] = np.nan

    # good_mask = {k: results[k] < threshold for k in sensor_dict}
    return sensor_dict, np.array(list(results.values())) < threshold