from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
from typing import Any, Dict, List, Sequence, Tuple

from .hr import get_near_baseline_segments, _plot_hr_with_segments, _plot_hr_histogram
from .actography import _plot_actography

logger = logging.getLogger(__name__)


def _plot_average_heartbeat(
    mean_epochs,
    std_epochs,
    time_axis,
    title="",
    ax=None,
):
    """Reusable plotting helper for one averaged heartbeat panel."""
    if ax is None:
        fig = plt.figure(figsize=(7.11, 3), dpi=300)
        ax = fig.gca()
        show_plot = True
    else:
        show_plot = False

    if mean_epochs.ndim > 1 and mean_epochs.shape[1] == 3:
        labels = ["x", "y", "z"]
        colors = ["C0", "C1", "C2"]
        color_fill = ["C0", "C1", "C2"]
        magnitude = np.linalg.norm(mean_epochs, axis=-1)
        handles = [plt.Line2D([0], [0], color=colors[i], label=labels[i]) for i in range(3)]
        handles.append(plt.Line2D([0], [0], color="k", linestyle=":", label="Magnitude"))
        linewidth = 1
    else:
        labels = None
        n_channels = mean_epochs.shape[1] if mean_epochs.ndim > 1 else 1
        colors = ["k"] * n_channels
        color_fill = ["grey"] * n_channels
        magnitude = None
        handles = [
            plt.Line2D([0], [0], color="k", label="Mean"),
            plt.Polygon([[0, 0], [0, 0], [0, 0]], color="grey", alpha=0.3, label="Standard Deviation"),
        ]
        linewidth = 0.5

    time_axis_ms = time_axis * 1000
    ax.hlines(0, time_axis_ms[0], time_axis_ms[-1], color="k", linestyle="--")

    if mean_epochs.ndim > 1:
        for i in range(mean_epochs.shape[1]):
            line_label = labels[i] if labels is not None else None
            ax.plot(time_axis_ms, mean_epochs[:, i], linewidth=linewidth, color=colors[i], label=line_label)
            ax.fill_between(
                time_axis_ms,
                mean_epochs[:, i] - std_epochs[:, i],
                mean_epochs[:, i] + std_epochs[:, i],
                alpha=0.2,
                color=color_fill[i],
            )
    else:
        ax.plot(time_axis_ms, mean_epochs, linewidth=1, color="k")
        ax.fill_between(
            time_axis_ms,
            mean_epochs - std_epochs,
            mean_epochs + std_epochs,
            alpha=0.2,
            color="k",
        )

    if magnitude is not None:
        ax.plot(time_axis_ms, magnitude, color="k", linestyle=":", label="Magnitude")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.5)
    ax.legend(handles=handles, loc="lower left", ncol=len(handles))
    ax.set_title(title)

    if show_plot:
        plt.show()

    return ax

def plot_unified_fetal_analysis(segments, data):
    """
    Creates a unified 2x2 plot for fetal analysis:
    - Top left: Heart rate with selected segments
    - Bottom left: Actogram (shares x-axis with HR)
    - Right: HR histogram (spans both rows)

    Parameters
    ----------
    segments : list of tuples
        List of (count, start, end) segment tuples
    data : dict
        Dictionary containing analysis data:
        - 'heart_rate': Heart rate array
        - 'baseline_hr': Baseline heart rate value
        - 'bpm_threshold': BPM threshold used
        - 'vss': Actogram velocity values
        - 'time_axis': Time axis for actogram
        - 'detections': Movement detections

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], hspace=0.3, wspace=0.2)

    # Top left: HR with segments
    ax_hr = fig.add_subplot(gs[0, 0])
    _plot_hr_with_segments(
        data['heart_rate'],
        segments,
        baseline_hr=data['baseline_hr'],
        bpm_threshold=data['bpm_threshold'],
        title="Heart Rate with Selected Segments",
        ax=ax_hr
    )

    # Bottom left: Actogram (shares x-axis with HR)
    ax_acto = fig.add_subplot(gs[1, 0], sharex=ax_hr)
    _plot_actography(
        data['time_axis'],
        data['vss'],
        detection=data['detections'],
        title="Actogram",
        ylabel="Velocity",
        ax=ax_acto
    )

    # Right: HR histogram (spans both rows)
    ax_hist = fig.add_subplot(gs[:, 1])
    _plot_hr_histogram(
        data['heart_rate'],
        title="Heart Rate Distribution",
        xlabel="Heart Rate [BPM]",
        ax=ax_hist,
        baseline_hr=data['baseline_hr'],
        bpm_threshold=data['bpm_threshold']
    )

    plt.tight_layout()
    plt.show()

    return fig


def compute_segment_averages(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    segments: Sequence[Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    """Compute heartbeat averages per selected peak-index segment."""
    outputs: List[Dict[str, Any]] = []
    peaks = np.asarray(peaks)
    signal = np.asarray(signal)

    for count, start, end in segments:
        if end - start < 2:
            continue
        p0 = peaks[start]
        p1 = peaks[end]
        if p1 <= p0 or p1 > signal.shape[0]:
            continue

        rr_intervals = np.diff(peaks[start : end + 1])
        rr_intervals = rr_intervals[rr_intervals > 0]
        seg_mean_hr = (
            float(np.nanmean(60.0 * float(fs) / rr_intervals))
            if rr_intervals.size > 0
            else np.nan
        )

        local_peaks = peaks[start:end] - p0
        local_signal = signal[p0:p1]
        mean, std, time_axis = average_heartbeats(local_peaks, local_signal, fs)

        segment_meta = {
            "count": int(count),
            "start": int(start),
            "end": int(end),
            "start_sec": float(peaks[start]) / float(fs),
            "end_sec": float(peaks[end]) / float(fs),
            "mean_hr": seg_mean_hr,
        }
        outputs.append(
            {
                "mean": mean,
                "std": std,
                "time_axis": time_axis,
                "segment": segment_meta,
            }
        )
    return outputs


def get_excerpt(
    signal: np.ndarray,
    fs: float,
    seconds: float,
    start_seconds: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return a bounded time excerpt from signal."""
    signal = np.asarray(signal)
    n = max(1, int(seconds * fs))
    n = min(n, signal.shape[0])
    max_start = max(0, signal.shape[0] - n)
    start_idx = int(round(start_seconds * fs))
    start_idx = max(0, min(start_idx, max_start))
    end_idx = start_idx + n
    time_axis = np.arange(start_idx, end_idx) / fs
    return signal[start_idx:end_idx], time_axis, start_idx


def segment_spans_seconds(
    peaks: np.ndarray,
    segments: Sequence[Tuple[int, int, int]],
    fs_hz: float,
    min_time: float = 0.0,
    max_time: float = np.inf,
) -> List[Tuple[float, float]]:
    """Convert peak-index segments to bounded time spans in seconds."""
    spans: List[Tuple[float, float]] = []
    peaks = np.asarray(peaks)
    for _, start, end in segments:
        if start >= len(peaks) or end >= len(peaks):
            continue
        t0 = float(peaks[start]) / fs_hz
        t1 = float(peaks[end]) / fs_hz
        if t0 > max_time or t1 < min_time:
            continue
        spans.append((max(min_time, t0), min(max_time, t1)))
    return spans


def find_segments_and_average_heartbeats(peaks, signal, fs, plot_k_segments=3, unified_plot=True, **kwargs):
    """
    Finds segments of near-baseline heart rate and averages heartbeats within those segments.

    Parameters
    ----------
    peaks : (n_peaks,)
        Sample indices of detected peaks
    signal : (n_samples, n_channels)
        The input signal from which to extract heartbeats
    fs : float
        Sampling frequency of the signal (in Hz)
    plot_k_segments : int
        Number of hr segments to plot individually. Defaults to 3. True will plot all segments, False will plot none.
    unified_plot : bool
        If True, creates a unified 2x2 plot showing HR, actogram, and histogram together.
        Overrides individual plots. Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to get_near_baseline_segments

    Returns
    -------
    avgs : list of tuples
        List of (mean_epochs, std_epochs, time_axis) for each segment
    segments : list of tuples
        List of (count, start, end) for each near-baseline segment
    """
    
    if unified_plot:
        # Get segments and all computed data
        segments, data = get_near_baseline_segments(
            peaks, fs, signal=signal, plot=False, return_data=True, **kwargs
        )

        # Create unified plot using the returned data
        plot_unified_fetal_analysis(segments, data)
    else:
        # Just get segments (optionally with individual plots)
        segments = get_near_baseline_segments(
            peaks, fs, signal=signal, plot=True, **kwargs
        )

    # Average heartbeats for each segment
    avgs = []
    for i, (count, start, end) in enumerate(segments):
        mean_epochs, std_epochs, time_axis = average_heartbeats(
            peaks[start:end] - peaks[start],
            signal[peaks[start] : peaks[end]],
            fs,
            plot=(plot_k_segments is True or (isinstance(plot_k_segments, int) and i < plot_k_segments)),
            title=f"Averaged Heartbeat ({peaks[start]/fs:.2f}s - {peaks[end]/fs:.2f}s, {count} Beats)",
        )
        avgs.append((mean_epochs, std_epochs, time_axis))
    return avgs, segments


def extract_epochs(signal, peaks, fs, ratio_pre=0.5, interval=2.5):
    """
    Extracts epochs from the given signal centered around the given peaks.

    Parameters
    ----------
    signal : (n_samples, n_channels)
        The input signal from which to extract epochs.
    peaks : (n_peaks,)
        Sample indices of detected peaks around which to extract epochs.
    fs : float
        Sampling frequency of the signal (in Hz).
    ratio_pre : float
        Ratio of the epoch interval to be placed before the peak. Must be between 0 and 1. Defaults to 0.5 (i.e., symmetric epochs).
    interval : float
        Epoch length as a multiple of the mean RR interval (mean inter-peak
        spacing), *not* in seconds: ``win = interval * mean(diff(peaks))``
        samples. Defaults to 2.5 (≈2.5 cardiac cycles). The epoch therefore
        scales with the detected heart rate.

    Returns
    -------
    epochs : (n_valid_peaks, n_epoch_samples, n_channels)
        Extracted epochs centered around valid peaks.
    time_axis : (n_epoch_samples,)
        Time axis for the epochs, centered around the peak (0 seconds at the peak).
    """

    # Ensure peaks is an array
    peaks = np.asarray(peaks)

    # If there are too few peaks to compute a reliable RR mean, return
    # an empty epochs array together with a sensible time axis. This
    # avoids subsequent ``mean`` calls on empty arrays which raise
    # numpy runtime warnings.
    if peaks.size < 2:
        # Degenerate case: fewer than 2 peaks means the RR interval is
        # undefined, so the usual RR-scaled window cannot be formed. Fall back
        # to ``interval`` seconds purely to return a sensibly-shaped empty array.
        n_epoch_samples = max(1, int(round(interval * fs)))
        n_channels = signal.shape[1] if signal.ndim > 1 else 1
        # centered time axis for the epoch
        time_axis = (np.arange(n_epoch_samples) - int(round(ratio_pre * n_epoch_samples))) / fs
        # return zero epochs (no valid peaks) so caller can handle empty-case
        epochs = np.empty((0, n_epoch_samples, n_channels))
        return epochs, time_axis

    # 1. Pre-calculate constant window boundaries
    win = interval * np.diff(peaks).mean()
    left_offset = int(ratio_pre * win)
    right_offset = int((1 - ratio_pre) * win)

    # 2. Vectorized check for valid peaks
    valid_mask = (peaks >= left_offset) & (peaks <= len(signal) - right_offset)
    valid_peaks = peaks[valid_mask]

    # 3. Extract epochs
    epochs = np.array([signal[p - left_offset : p + right_offset] for p in valid_peaks])

    # remove epochs with NaN values (if any)
    if signal.ndim == 1:
        epochs = epochs[:, :, np.newaxis]
    valid_epochs_mask = ~np.isnan(epochs).any(axis=(1, 2))
    epochs = epochs[valid_epochs_mask]

    if signal.ndim == 1:
        epochs = epochs.squeeze(-1)

    # Time axis calculation
    # If no epochs were extracted, build the time axis from the same RR-scaled
    # window (left_offset/right_offset) so the shape matches a real epoch.
    if epochs.size == 0:
        n_epoch_samples = max(1, left_offset + right_offset)
        time_axis = (np.arange(n_epoch_samples) - left_offset) / fs
    else:
        time_axis = (np.arange(epochs.shape[1]) - left_offset) / fs

    return epochs, time_axis

def average_heartbeats(peaks, signal, fs, ratio_pre=0.5, interval=2.5, plot=False, title=""):
    """
    Averages heartbeats from the given signal based on detected peaks.

    Parameters
    ----------
    peaks : (n_peaks,)
        Sample indices of detected peaks around which to extract and average heartbeats.
    signal : (n_samples, n_channels)
        The input signal from which to extract and average heartbeats.
    fs : float
        Sampling frequency of the signal (in Hz).
    ratio_pre : float
        Ratio of the epoch interval to be placed before the peak. Must be between 0 and 1. Defaults to 0.5 (i.e., symmetric epochs).
    interval : float
        Epoch length as a multiple of the mean RR interval (mean inter-peak
        spacing), *not* in seconds: ``win = interval * mean(diff(peaks))``
        samples. Defaults to 2.5 (≈2.5 cardiac cycles). The epoch therefore
        scales with the detected heart rate.
    plot : bool
        If True, plots the average heartbeat with standard deviation shading. Defaults to False.

    Returns
    -------
    mean_epochs : (n_epoch_samples, n_channels)
        The average heartbeat across all valid epochs.
    std_epochs : (n_epoch_samples, n_channels)
        The standard deviation of the heartbeat across all valid epochs.
    time_axis : (n_epoch_samples,)
        Time axis for the epochs, centered around the peak (0 seconds at the peak).
    """
    epochs, time_axis = extract_epochs(signal, peaks, fs, ratio_pre, interval)

    # If no valid epochs were extracted, return NaN-filled arrays with a
    # sensible epoch length (based on the requested interval) to avoid
    # ``mean`` on empty arrays and to keep downstream code shape-stable.
    if epochs.size == 0:
        n_channels = signal.shape[1] if signal.ndim > 1 else 1
        # Match the (RR-scaled) length of the time axis from extract_epochs.
        if time_axis is not None and len(time_axis) > 0:
            n_epoch_samples = len(time_axis)
        else:
            n_epoch_samples = max(1, int(round(interval * fs)))
            time_axis = (np.arange(n_epoch_samples) - int(round(ratio_pre * n_epoch_samples))) / fs
        mean_epochs = np.full((n_epoch_samples, n_channels), np.nan)
        std_epochs = np.full((n_epoch_samples, n_channels), np.nan)
    else:
        # Average and standard deviation across epochs
        mean_epochs = epochs.mean(axis=0)
        std_epochs = epochs.std(axis=0)

    if plot:
        _plot_average_heartbeat(mean_epochs, std_epochs, time_axis, title=title)

    return mean_epochs, std_epochs, time_axis