import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import logging
from scipy.ndimage import uniform_filter1d
import warnings

from .actography import compute_actogram

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_peaks(signal, fs):
    """
    Cleans an ECG-like signal and identifies R-peak indices.
    """
    # Invert signal
    if signal.shape[0] > 2 * fs:
        signal, _ = nk.ecg_invert(signal, sampling_rate=fs)

    # Clean using the 'vg' method
    signal_clean = nk.ecg_clean(signal, sampling_rate=fs, method="vg")

    # Extract peaks
    peaks_dict = nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")
    return peaks_dict["ECG_R_Peaks"]


def compute_hr(signal, fs, comps=None, label="", plot=True, fetal=True):
    """
    Overall orchestrator to compute Heart Rate, identify peaks,
    and optionally visualize the results.
    """
    if comps is None:
        comps = np.arange(signal.shape[1]) if signal.ndim > 1 else None

    # 1. Identify best peaks
    peaks, selected_idx = _find_best_hr_estimate(signal, comps, fs//2 if fetal else fs)

    # 2. Calculate intervals and heart rates
    # Formula: HR = 60 / (RR_interval_in_seconds)
    all_intervals = np.diff(peaks) / fs
    heart_rate = 60 / all_intervals
    avg_hr = np.mean(heart_rate)

    # 3. Plotting
    if plot:
        _plot_hr(
            heart_rate, title=f"{label} Heart Rate Analysis (Avg: {avg_hr:.2f} BPM)"
        )
        _plot_hr_histogram(heart_rate, title=f"{label} Heart Rate Distribution")

    return heart_rate, peaks


def detect_hr_outlier(peaks, fs, win=4, threshold=30, plot=False):
    """
    Detects heart rate (HR) outliers based the percent change (PC) of the interbeat interval deviation to the mean of
    the previous intervals.
    Parameters:
        peaks (array-like): Indices of the detected peaks in the signal.
        fs (float): Sampling frequency of the data in Hz.
        win (int, optional): Window size for the running mean. Default is 4.
        threshold (float, optional): Percentage deviation threshold to classify HR as an outlier. Default is 30.
        plot (bool, optional): If True, plots the heart rate and its running mean. Default is False.
    Returns:
        numpy.ndarray: A boolean array indicating whether each heart rate is an outlier (True) or not (False).

    References:
    - Kemper, K., Hamilton, C. & Atkinson, M. Heart Rate Variability: Impact of Differences in Outlier Identification and Management Strategies on Common Measures in Three Clinical Populations. Pediatr Res 62, 337–342 (2007). https://doi.org/10.1203/PDR.0b013e318123fbcc
    """

    hr = 60 * fs / np.diff(peaks)

    # Compute backward-looking running mean (mean of PREVIOUS win intervals, NOT including current)
    # This ensures outliers don't dilute their own detection
    running_mean = np.zeros_like(hr)
    for i in range(len(hr)):
        if i == 0:
            # First beat: no previous values, use global mean
            running_mean[i] = np.mean(hr)
        elif i < win:
            # Not enough previous values: use all previous
            running_mean[i] = np.mean(hr[:i])
        else:
            # Sufficient history: use previous win values
            running_mean[i] = np.mean(hr[i - win : i])

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(7.11, 3), sharex=True)
        axs[0].plot(hr)
        axs[0].plot(running_mean)
        axs[0].legend(["Heart Rate", "Running Mean (previous)"])
        axs[0].set_ylabel("Heart Rate [bpm]")
        axs[0].set_xlabel("Beats")

        pct_deviation = np.absolute(hr - running_mean) / running_mean * 100
        axs[1].plot(pct_deviation)
        axs[1].hlines(threshold, 0, len(hr), color="k", linestyle="--")
        axs[1].legend(["Percentage Deviation", "Threshold"])
        axs[1].set_ylabel("Mean Deviation [%]")
        axs[1].set_xlabel("Beats")
        plt.show()

    # Compute percentage deviation and detect outliers
    pct_deviation = np.absolute(hr - running_mean) / running_mean * 100
    outliers = pct_deviation > threshold

    # Pad to length len(peaks) to maintain compatibility with existing code
    # outliers[i] indicates if HR between peaks[i] and peaks[i+1] is an outlier
    # outliers[len(peaks)-1] is padded (not meaningful, rarely used)
    outliers_padded = np.pad(
        outliers, pad_width=(0, 1), mode="constant", constant_values=False
    )

    return outliers_padded


def get_hr_baseline(hr, stable_and_quiescent, percentile=25, filter_length=60):
    """Fills outliers with median and applies a rolling mean to get a stable baseline estimate."""
    hr_filled = hr.copy()
    hr_filled[~stable_and_quiescent] = np.median(hr[stable_and_quiescent])
    hr_smoothed = uniform_filter1d(hr_filled, size=filter_length, mode="nearest")
    baseline_hr = float(np.percentile(hr_smoothed[stable_and_quiescent], percentile))
    return baseline_hr

from matplotlib.legend_handler import HandlerBase

# Define a custom handler that draws text instead of a line/marker
class TextHandler(HandlerBase):
    def __init__(self, text, color="black", **kwargs):
        self.text = text
        self.color = color
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create a text artist inside the legend icon bounding box
        tx = plt.Text(width/2, height/2, self.text, color=self.color,
                      ha="center", va="center", fontsize=fontsize)
        return [tx]

def get_near_baseline_segments(
    peaks,
    fs,
    detections=None,
    signal=None,
    bpm_threshold=5,
    min_segment_length=20,
    acto_min_beat_gap=10,
    outlier_win=10,
    outlier_threshold=5,
    plot=True,
    return_data=False,
):
    """Identifies segments of heart rate that are near the baseline, during stable and quiescent periods.

     Parameters:
        peaks (array-like): Indices of the detected peaks in the signal.
        fs (float): Sampling frequency of the data in Hz.
        detections (array-like, optional): Boolean array indicating detected movement periods. If None, it will be computed from the signal.
        signal (array-like, optional): Multichannel signal data used to compute movement detections if 'detections' is not provided.

        Settings controlling segment selection:
            bpm_threshold (float, optional): Maximum allowed deviation from baseline heart rate in BPM to be considered "near baseline". Default is 5 BPM.
            min_segment_length (int, optional): Minimum number of consecutive beats in a segment to be included in the output. Default is 20 beats.

        Settings controlling the labeling of stable and quiescent periods using actography as a proxy for movement and heartrate outlier detection to exclude unstable periods:
            acto_min_beat_gap (int, optional): Minimum gap in beats for merging detections when computing actogram. Default is 10 beats.
            outlier_win (int, optional): Window size in beats for detecting heart rate outliers. Default is 10 beats.
            outlier_threshold (float, optional): Percentage deviation threshold for classifying heart rate intervals as outliers. Default is 5%.
        plot (bool, optional): If True, plots the heart rate and highlights near-baseline segments. Default is True.
        return_data (bool, optional): If True, returns additional computed data (heart_rate, baseline_hr, vss, time_axis, detections). Default is False.

    Returns:
        segments : list of tuples (segment_length, start_index, end_index)
            A list of segments where each tuple contains the length of the segment in beats,
            and the start and end indices of the segment in the 'peaks' array.
            The list is sorted in descending order by segment length.
        data : dict (only if return_data=True)
            Dictionary containing:
                - 'heart_rate': Heart rate array
                - 'baseline_hr': Baseline heart rate value
                - 'bpm_threshold': BPM threshold used
                - 'vss': Actogram velocity values (if signal provided)
                - 'time_axis': Time axis for actogram (if signal provided)
                - 'detections': Movement detections (if signal provided)

    References:
        - Strand, Sarah A., Janette F. Strasburger, and Ronald T. Wakai. 'Fetal Magnetocardiogram Waveform Characteristics'. Physiological Measurement 40, no. 3 (2019): 035002. https://doi.org/10.1088/1361-6579/ab0a2c.
    """
    # Detects outliers based the percent change (PC) of the interbeat interval deviation to the mean of the previous intervals
    stable_hr = ~detect_hr_outlier(
        peaks, fs, win=outlier_win, threshold=outlier_threshold, plot=False
    )[:-1]

    # Store actogram data if computed
    vss, time_axis = None, None

    if detections is None:
        if signal is None:
            detections = np.zeros_like(stable_hr, dtype=bool)
            warnings.warn(
                "No movement detections or signal provided. Assuming all beats are quiescent."
            )
        else:
            vss, time_axis, detections = compute_actogram(
                signal, peaks, fs, plot=plot, min_gap_beats=acto_min_beat_gap
            )

    quiescent = ~detections
    stable_and_quiescent = stable_hr & quiescent

    # Smooth HR by filling outliers with median and applying a rolling mean to get a more stable baseline estimate
    heartRate = 60 * fs / np.diff(peaks)
    baseline_hr = get_hr_baseline(heartRate, stable_and_quiescent)

    # selection mask for beats near the baseline (within 5 BPM) and during stable & quiescent periods
    near_baseline = (
        np.abs(heartRate - baseline_hr) <= bpm_threshold
    ) & stable_and_quiescent

    # Get consecutive segments of near-baseline beats, filter by minimum length, and sort by length
    segments = _get_consecutive_segments(near_baseline, min_length=min_segment_length)

    if plot:
        fig = plt.figure(figsize=(7.11, 3), dpi=300)
        x_positions = np.cumsum(60 / heartRate)
        plt.plot(
            x_positions,
            heartRate,
            ".-",
            label="Heart Rate",
            color="k",
            fillstyle="none",
            linewidth=0.75,
        )
        plt.axhline(
            baseline_hr,
            color="red",
            linestyle="--",
            label=f"Baseline HR: {baseline_hr:.2f} BPM",
            zorder=6
        )
        plt.scatter(
            x_positions[near_baseline],
            heartRate[near_baseline],
            color="green",
            label=f"Near Baseline (+/-{bpm_threshold} BPM)",
            marker="o",
            s=3,
            edgecolor="b",
            zorder=5,
        )

        for i, (counts, start, end) in enumerate(segments):
            plt.axvspan(
                x_positions[start],
                x_positions[end],
                color=f"C{i}",
                alpha=0.3,
                label=f"Consecutive beats" if i == 0 else None,
            )

            plt.text(
                x_positions[start] + (x_positions[end] - x_positions[start]) / 2,
                np.max(heartRate) * 0.95,
                f"{counts}",
                color=f"C{i}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        handles, labels = plt.gca().get_legend_handles_labels()
        text_key = object()
        handles.append(text_key)
        labels.append("Number of beats")

        # 4. Pass the custom handler map to plt.legend
        plt.legend(
            handles=handles, 
            labels=labels, 
            handler_map={text_key: TextHandler(segments[0][0], color="C0")},
            loc="lower right" 
        )

        plt.xlabel("Time [s]")
        plt.ylabel("Heart Rate [BPM]")
        plt.grid(True, linestyle="--", alpha=0.9)
        plt.minorticks_on()
        plt.grid(which="minor", linestyle=":", alpha=0.5)
        plt.show()

    if return_data:
        data = {
            'heart_rate': heartRate,
            'baseline_hr': baseline_hr,
            'bpm_threshold': bpm_threshold,
        }
        if vss is not None:
            data['vss'] = vss
            data['time_axis'] = time_axis
            data['detections'] = detections
        return segments, data

    return segments


def split_ectopic_beats(
    peaks,
    fs,
    win=10,
    threshold=15,
    signal=None,
    morph_threshold=4.0,
    ratio_pre=0.5,
    interval=1.0,
    plot=True,
    return_data=False,
):
    """Split R-peaks into NORMAL and ECTOPIC/OUTLIER beats for type-aware averaging and segmentation.

    Averaging an ectopic beat is ill-posed: it does not belong in the normal average (different
    morphology), and a near-baseline segmentation (`get_near_baseline_segments`) excludes it as an
    outlier, so forcing it in blends beat types. This function labels each beat so the normal beats can
    be averaged / segmented cleanly while the ectopic beats are kept aside for separate inspection.

    Detection combines the beat-to-beat RHYTHM (reusing `detect_hr_outlier`: percent change of the R-R
    interval vs a backward running mean; a premature beat is a short R-R / HR spike, a pause or dropped
    beat is a long R-R / HR dip) and, optionally, beat MORPHOLOGY (if `signal` is given: a beat whose
    waveform is far from the normal-beat template by a robust z-score is also flagged, catching ectopics
    that do not perturb the rhythm).

    Parameters
    ----------
    peaks : (n_peaks,) array
        R-peak sample indices.
    fs : float
        Sampling frequency [Hz].
    win : int, optional
        Backward running-mean window [beats] for the rhythm outlier test (`detect_hr_outlier`). Default 10.
    threshold : float, optional
        Percent R-R deviation to flag a rhythm outlier. Default 15.
    signal : (n_samples, n_channels) array, optional
        If given, adds morphology-based flagging on the beats epoched from this signal.
    morph_threshold : float, optional
        Robust z-score of the beat-to-template distance above which a beat is a morphology outlier. Default 4.
    ratio_pre, interval : float, optional
        Epoching parameters for the morphology path (passed to `extract_epochs`).
    plot : bool, optional
        If True, plot the fetal HR over time with the flagged beats annotated (default True).
    return_data : bool, optional
        If True, also return a dict with the heart rate, baseline and per-beat labels.

    Returns
    -------
    normal_idx : (n_normal,) int array
        Indices into `peaks` of beats safe to average / segment.
    ectopic_idx : (n_ectopic,) int array
        Indices of flagged beats (premature, pause/dropped, or morphology outlier). Note both endpoints of
        each rhythm-outlier interval are excluded, so the normal set is conservative (safe for averaging).
    labels : (n_peaks,) int array
        Per-beat label: 0 normal, +1 premature, -1 pause/dropped, 2 morphology-only outlier.
    data : dict, optional
        Returned only if ``return_data=True`` (heart_rate, baseline_hr, labels, counts).
    """
    peaks = np.asarray(peaks)
    n = len(peaks)
    hr = 60.0 * fs / np.diff(peaks)                                     # per-interval HR, len n-1
    iv_out = detect_hr_outlier(peaks, fs, win=win, threshold=threshold, plot=False)[:-1]
    baseline = float(np.median(hr))

    # A rhythm-outlier interval i (between peaks i and i+1) implicates the LATER beat i+1.
    labels = np.zeros(n, dtype=int)
    labels[np.where(iv_out & (hr > baseline))[0] + 1] = 1               # short R-R -> premature
    labels[np.where(iv_out & (hr <= baseline))[0] + 1] = -1            # long  R-R -> pause / dropped

    if signal is not None:                                             # optional morphology refinement
        from .heartbeat_averaging import extract_epochs               # local import avoids circular import
        ep, _ = extract_epochs(signal, peaks, fs, ratio_pre=ratio_pre, interval=interval)
        m = ep.shape[0]
        base = labels[:m] == 0
        templ = np.median(ep[base] if base.any() else ep, axis=0)
        d = np.linalg.norm((ep - templ).reshape(m, -1), axis=1)
        dz = (d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12)
        morph = np.where(dz > morph_threshold)[0]
        labels[morph[labels[morph] == 0]] = 2                          # only if not already a rhythm outlier

    excl = labels != 0
    excl[np.where(iv_out)[0]] = True                                    # also drop the earlier endpoint of each outlier interval
    normal_idx = np.where(~excl)[0]
    ectopic_idx = np.where(excl)[0]

    if plot:
        _plot_ectopic_beats(hr, labels, baseline, win, threshold)

    if return_data:
        data = {
            "heart_rate": hr,
            "baseline_hr": baseline,
            "labels": labels,
            "n_premature": int((labels == 1).sum()),
            "n_pause": int((labels == -1).sum()),
            "n_morph": int((labels == 2).sum()),
        }
        return normal_idx, ectopic_idx, labels, data
    return normal_idx, ectopic_idx, labels


def _plot_ectopic_beats(hr, labels, baseline_hr, win, threshold):
    """Fetal HR over time with the flagged beats annotated (premature / pause / morphology)."""
    x = np.cumsum(60.0 / hr)                                            # time [s]; beat j sits at x[j-1]
    fig = plt.figure(figsize=(7.11, 3), dpi=300)
    ax = fig.gca()
    ax.plot(x, hr, ".-", color="k", fillstyle="none", linewidth=0.75, label="Heart Rate")
    ax.axhline(baseline_hr, color="red", linestyle="--", linewidth=1, label=f"Median HR: {baseline_hr:.1f} BPM")
    for code, marker, color, name, filled in [(1, "v", "tab:red", "premature", True),
                                              (-1, "^", "tab:orange", "pause / dropped", True),
                                              (2, "x", "tab:purple", "morphology", False)]:
        j = np.where(labels == code)[0]
        j = j[j >= 1]                                                  # beat j -> interval j-1
        if len(j):
            kw = dict(edgecolor="k", linewidths=0.4) if filled else {}
            ax.scatter(x[j - 1], hr[j - 1], marker=marker, color=color, s=40, zorder=5,
                       label=f"{name} ({len(j)})", **kw)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heart Rate [BPM]")
    ax.set_title(f"Ectopic-beat detection (win={win}, threshold={threshold}%)")
    ax.grid(True, linestyle="--", alpha=0.9)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
    return ax


def _get_consecutive_segments(bool_array, min_length=10):
    """Finds consecutive True segments in a boolean array, filters by minimum length, and returns sorted segments.

    Parameters
    ----------
    bool_array : array-like
        Input boolean array to analyze for consecutive True segments.
    min_length : int, optional
        Minimum length of consecutive True segments to include in the output (default is 10).

    Returns
    -------
    list of tuples (segment_length, segment_slice)
        A list of tuples, each containing the length of the segment and the segment's start and end indices. The list is sorted in descending order by segment length.
    """
    # Convert boolean to int and find transitions
    padded = np.pad(bool_array.astype(int), (1, 1), "constant")
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    ends = np.clip(ends, 0, len(bool_array) - 1)
    counts = ends - starts

    # Zip, filter, and sort in one go
    sorted_segments = sorted(
        [(c, s, e) for c, s, e in zip(counts, starts, ends) if c >= min_length],
        key=lambda x: x[0],
        reverse=True,
    )
    return sorted_segments


def _find_best_hr_estimate(signal, comps, fs):
    """
    Evaluates multiple components and selects the one with the
    most regular rhythm (lowest SDNN).
    """
    if comps is not None and signal.ndim > 1 and len(comps) > 1:
        logger.debug("Selecting the Fetal Component with lowest SDNN")

        # Calculate SDNN for each provided component index
        sdnn_values = []
        for f in comps:
            peaks = detect_peaks(signal[:, f], fs)
            sdnn_values.append(np.std(np.diff(peaks)))

        best_comp_idx = comps[np.argmin(sdnn_values)]
        peaks = detect_peaks(signal[:, best_comp_idx], fs)

        logger.debug(
            f"Selected component {best_comp_idx} with SDNN {np.min(sdnn_values)}"
        )
        return peaks, best_comp_idx

    # Fallback for single component or 1D signal
    target_signal = signal[:, comps[0]] if comps is not None else signal
    return detect_peaks(target_signal, fs), comps


def _plot_hr(heart_rate, title="", ylabel="Heart Rate [BPM]", ax=None, label=None, color="tab:green"):
    """Internal helper for visualization."""
    if ax is None:
        fig = plt.figure(figsize=(7.11, 3), dpi=300)
        ax = fig.gca()
        show_plot = True
    else:
        show_plot = False

    # Main Line
    x_positions = np.cumsum(60 / heart_rate)
    ax.plot(
        x_positions,
        heart_rate,
        ".-",
        label=label,
        color="k",
        fillstyle="none",
        linewidth=0.75,
    )

    # Highlight points
    ax.plot(x_positions, heart_rate, ".", color=color, markersize=2)

    # Styling
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.9)

    # x ticks every 10 seconds
    ax.set_xticks(np.arange(0, x_positions[-1], 15))

    ax.legend()

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax


def _plot_hr_with_segments(heart_rate, segments, baseline_hr=None, bpm_threshold=None, title="", ylabel="Heart Rate [BPM]", ax=None):
    """Plot heart rate with near-baseline segments highlighted."""
    if ax is None:
        fig = plt.figure(figsize=(7.11, 3), dpi=300)
        ax = fig.gca()
        show_plot = True
    else:
        show_plot = False

    x_positions = np.cumsum(60 / heart_rate)
    ax.plot(
        x_positions,
        heart_rate,
        ".-",
        label="Heart Rate",
        color="k",
        fillstyle="none",
        linewidth=0.75,
    )

    if baseline_hr is not None:
        ax.axhline(
            baseline_hr,
            color="red",
            linestyle="--",
            label=f"Baseline HR: {baseline_hr:.2f} BPM",
        )

    # Find near-baseline mask if segments are provided
    if segments:
        near_baseline = np.zeros(len(heart_rate), dtype=bool)
        for counts, start, end in segments:
            near_baseline[start:end+1] = True

        ax.scatter(
            x_positions[near_baseline],
            heart_rate[near_baseline],
            color="green",
            label=f"Near Baseline (+/-{bpm_threshold} BPM)" if baseline_hr is not None else "Near Baseline",
            marker="o",
            s=3,
            edgecolor="b",
            zorder=5,
        )

        for i, (counts, start, end) in enumerate(segments):
            ax.axvspan(
                x_positions[start],
                x_positions[end],
                color=f"C{i}",
                alpha=0.3,
                label=f"Consecutive beats" if i == 0 else None,
            )
            ax.text(
                x_positions[start] + (x_positions[end] - x_positions[start]) / 2,
                np.max(heart_rate)*.95,
                f"{counts}",
                color=f"C{i}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    text_key = object()
    handles.append(text_key)
    labels.append("Number of beats")

    # Pass the custom handler map to plt.legend
    plt.legend(
        handles=handles, 
        labels=labels, 
        handler_map={text_key: TextHandler(segments[0][0], color="C0")},
        loc="lower right" 
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.9)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.5)

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax


def _plot_hr_histogram(heart_rate, title="", xlabel="Heart Rate [BPM]", ax=None, baseline_hr=None, bpm_threshold=None, bin_width=2):
    if ax is None:
        fig = plt.figure(figsize=(7.11, 2), dpi=300)
        ax = fig.gca()
        show_plot = True
    else:
        show_plot = False

    bins = np.arange(
        np.floor(heart_rate.min()) - 0.5,
        np.ceil(heart_rate.max()) + 0.5 + bin_width,
        bin_width,
    )

    counts, _, _ = ax.hist(
        heart_rate, bins=bins, color="tab:blue", alpha=0.7, edgecolor="k", zorder=1
    )

    # Visualize baseline ± threshold if provided
    if baseline_hr is not None:
        ax.axvline(baseline_hr, color="red", linestyle="--", linewidth=1.5, label=f"Baseline: {baseline_hr:.1f} BPM", zorder=3)
        if bpm_threshold is not None:
            ax.axvspan(
                baseline_hr - bpm_threshold,
                baseline_hr + bpm_threshold,
                color="red",
                alpha=0.1,
                label=f"±{bpm_threshold} BPM",
                zorder=2
            )
            ax.axvline(baseline_hr - bpm_threshold, color="red", linestyle=":", linewidth=1, alpha=0.5, zorder=2)
            ax.axvline(baseline_hr + bpm_threshold, color="red", linestyle=":", linewidth=1, alpha=0.5, zorder=2)

    # Align ticks with bin centers (integers) - but show fewer to reduce clutter
    centers = bins[:-1] + bin_width / 2
    hr_range = heart_rate.max() - heart_rate.min()

    # Determine tick spacing based on range
    if hr_range > 50:
        tick_spacing = 10
    elif hr_range > 20:
        tick_spacing = 5
    else:
        tick_spacing = 2

    # Find ticks that are multiples of tick_spacing, with robust fallback.
    tick_values = centers[np.mod(centers.astype(int), tick_spacing) == 0]
    if len(tick_values) == 0:
        tick_min = int(np.floor(heart_rate.min()))
        tick_max = int(np.ceil(heart_rate.max()))
        tick_values = np.arange(tick_min, tick_max + tick_spacing, tick_spacing)
        if len(tick_values) == 0:
            tick_values = np.array([tick_min, tick_max])
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{int(t)}" for t in tick_values], rotation=0)
    ax.tick_params(axis="x", labelbottom=True)

    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Intervals")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.9, zorder=0)


    if baseline_hr is not None:
        ax.legend(loc="upper left", ncol=2)

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax
