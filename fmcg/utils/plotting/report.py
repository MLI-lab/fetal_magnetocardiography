import os

import matplotlib.backends.backend_pdf
import numpy as np
from matplotlib import pyplot as plt

from fmcg.utils.plotting.plot_utils import logger

# ----------------------------- UTILS -----------------------------


def _safe_finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _format_segment_details(segment):
    """Build a compact, human-readable segment descriptor for plot titles."""
    if not isinstance(segment, dict):
        return None

    parts = []

    start_sec = _safe_finite_float(segment.get("start_sec"))
    end_sec = _safe_finite_float(segment.get("end_sec"))
    if start_sec is not None and end_sec is not None:
        parts.append(f"{start_sec:.1f}s to {end_sec:.1f}s")
    else:
        try:
            start_idx = int(segment.get("start"))
            end_idx = int(segment.get("end"))
            parts.append(f"idx={start_idx}:{end_idx}")
        except (TypeError, ValueError):
            pass

    try:
        beat_count = int(segment.get("count"))
        parts.append(f"{beat_count} beats")
    except (TypeError, ValueError):
        pass

    mean_hr = _safe_finite_float(segment.get("mean_hr"))
    if mean_hr is not None:
        parts.append(f"{mean_hr:.1f} bpm")

    return " | ".join(parts) if parts else None


def _create_segment_page(max_rows, width, height):
    """Create a standard page layout and normalize axes to a 1D array."""
    fig, axes = plt.subplots(
        nrows=max_rows,
        ncols=1,
        figsize=(width, height),
        sharex=False,
    )
    return fig, np.atleast_1d(axes)


def _disable_unused_axes(axes, page_idx, num_segments):
    """Hide rows that exceed the number of available segments."""
    for i, ax in enumerate(axes):
        if page_idx + i >= num_segments:
            ax.axis("off")


def _plot_segment_outliers(ax, time_, signal, peaks, outlier_mask, start, end):
    """Plot outlier shading for peaks in the current segment.

    Supports both sample-indexed masks (len == len(time_)) and peak-indexed masks
    (len == len(peaks) or len == len(peaks)-1 from detect_hr_outlier internals).
    """
    peaks = np.asarray(peaks)
    outlier_mask = np.asarray(outlier_mask).astype(bool)

    peak_positions = np.where((peaks >= start) & (peaks < end))[0]
    if peak_positions.size == 0 or outlier_mask.size == 0:
        return

    peaks_in_segment = peaks[peak_positions]

    if outlier_mask.size == len(time_):
        valid = (peaks_in_segment >= 0) & (peaks_in_segment < outlier_mask.size)
        peaks_in_segment = peaks_in_segment[valid]
        if peaks_in_segment.size == 0:
            return
        outlier_for_peaks = outlier_mask[peaks_in_segment]
    elif outlier_mask.size == peaks.size:
        outlier_for_peaks = outlier_mask[peak_positions]
    elif outlier_mask.size + 1 == peaks.size:
        valid_positions = peak_positions[peak_positions < outlier_mask.size]
        if valid_positions.size == 0:
            return
        peaks_in_segment = peaks[valid_positions]
        outlier_for_peaks = outlier_mask[valid_positions]
    else:
        n = min(peaks_in_segment.size, outlier_mask.size)
        if n == 0:
            return
        peaks_in_segment = peaks_in_segment[:n]
        outlier_for_peaks = outlier_mask[:n]

    if not np.any(outlier_for_peaks):
        return

    ax.fill_between(
        time_[peaks_in_segment],
        np.nanmin(signal[start:end]),
        np.nanmax(signal[start:end]),
        where=outlier_for_peaks,
        alpha=0.25,
        zorder=0,
        interpolate=False,
        label="Outlier",
        color="grey",
    )


# ----------------------------- REPORTS -----------------------------

def plot_signal(x, y, xlabel="Time [s]", ylabel="Signal", color="k", label="Signal", title=None, savename=None):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(x, y, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="upper right") 
    if title:
        ax.set_title(title)
    if savename:
        fig.savefig(savename, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def report_hr(
    data_dict,
    heartbeats_dict,
    time_,
    output_dir,
    segment_duration=60,
):
    """
    Generates segmented heart rate (HR) report plots and saves them as a multi-page PDF.

    This function visualizes heart rate data over time, highlighting artifacts, outliers, and selected peaks
    for each segment. Each page of the PDF contains multiple time segments, with HR plots for each segment.
    Artifacts and outliers are shaded, and selected peaks are marked for visual inspection.

    Args:
        data_dict (dict): Dictionary containing HR data and associated metadata for each signal.
            Expected keys for each signal:
                - "hr": Heart rate array.
                - "peaks": Indices of detected peaks.
                - "outlier": Boolean mask for outlier peaks.
                - "artifacts" (optional): Boolean mask for artifact regions.
        heartbeats_dict (dict): Dictionary containing heartbeat peak selection information for each signal.
            Expected keys for each signal:
                - "M_x": Dictionary with either:
                    - "selected_peak_times": List of selected peak times (in seconds).
                    - "selected_peaks_mask": Boolean mask for selected peaks.
        time_ (np.ndarray): Array of time values corresponding to HR samples.
        output_dir (str): Directory path to save the generated PDF report.
        segment_duration (int, optional): Duration (in seconds) of each segment to plot per row. Default is 60.
    """
    # Create PDF backend
    pdf_path = os.path.join(output_dir, "3_hr_REPORT.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    # Constants for A4 size and layout
    A4_WIDTH, A4_HEIGHT = 8.27, 11.69  # inches
    ROW_HEIGHT = 2  # inches per row
    MAX_ROWS = int((A4_HEIGHT - 1) / ROW_HEIGHT)  # Leave 1 inch for title and margins

    # Calculate segment indices
    fs = int(1 / (time_[1] - time_[0]))  # Sampling frequency
    segment_samples = segment_duration * fs
    num_segments = int(np.ceil(len(time_) / segment_samples))

    for key in data_dict.keys():
        hr = data_dict[key]["hr"].copy()

        # For segmented processing, use artifacts if available
        if "artifacts" in data_dict[key]:
            artifact_idx = data_dict[key]["artifacts"].astype(int)

        for page_idx in range(0, num_segments, MAX_ROWS):
            fig, axes = _create_segment_page(MAX_ROWS, A4_WIDTH, A4_HEIGHT)

            # disable axes that are not used
            _disable_unused_axes(axes, page_idx, num_segments)

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], hr[start:end])

                # Handle peaks and outliers differently for segmented vs non-segmented processing
                peaks = data_dict[key]["peaks"]
                _plot_segment_outliers(
                    ax,
                    time_,
                    hr,
                    peaks,
                    data_dict[key]["outlier"],
                    start,
                    end,
                )

                if "artifacts" in data_dict[key]:
                    # Add artifact highlighting
                    if np.any(artifact_idx[start:end]):
                        ax.fill_between(
                            time_[start:end],
                            np.nanmin(hr[start:end]),
                            np.nanmax(hr[start:end]),
                            where=artifact_idx[start:end],
                            alpha=0.25,
                            zorder=0,
                            interpolate=False,
                            label="Artifact",
                            color="orange",
                        )

                if key in heartbeats_dict.keys() and "M_x" in heartbeats_dict[key]:
                    # Check if we have selected peak times (for segmented processing)
                    if "selected_peak_times" in heartbeats_dict[key]["M_x"]:
                        selected_peak_times = heartbeats_dict[key]["M_x"][
                            "selected_peak_times"
                        ]
                        if len(selected_peak_times) > 0:
                            # Convert times to indices in the current time window
                            selected_indices = []
                            for peak_time in selected_peak_times:
                                # Find closest time index in the current segment
                                time_idx = np.argmin(np.abs(time_ - peak_time))
                                if start <= time_idx < end:
                                    selected_indices.append(time_idx)

                            if len(selected_indices) > 0:
                                selected_indices = np.array(selected_indices)
                                ax.plot(
                                    time_[selected_indices],
                                    hr[selected_indices],
                                    "rx",
                                    label="Selected peaks",
                                )
                    # Fallback to mask-based approach (for non-segmented processing)
                    elif "selected_peaks_mask" in heartbeats_dict[key]["M_x"]:
                        mask = heartbeats_dict[key]["M_x"]["selected_peaks_mask"]
                        if len(mask) > 0 and len(peaks) > 1:
                            # Ensure mask length matches peaks length
                            mask_length = min(len(mask), len(peaks) - 1)
                            mask = mask[:mask_length]
                            selected_peaks = peaks[1 : mask_length + 1][mask]
                            selected_peaks = selected_peaks[
                                (selected_peaks >= start) & (selected_peaks < end)
                            ]
                            if len(selected_peaks) > 0:
                                ax.plot(
                                    time_[selected_peaks],
                                    hr[selected_peaks],
                                    "rx",
                                    label="Selected peaks",
                                )
                    # if data_dict[key].get("peak_selection_mask", None) is not None:

                    #     selected_peaks = peaks[1:][data_dict[key]["peak_selection_mask"]]
                    #     selected_peaks = selected_peaks[(selected_peaks >= start) & (selected_peaks < end)]
                    #     ax.plot(
                    #         time_[selected_peaks],
                    #         data_dict[key]["hr"][selected_peaks],
                    #         "rx",
                    #     )

                ax.legend(loc="upper right")
                ax.set_title(
                    f"{key.capitalize()}: {time_[start]:.1f}s to {time_[end - 1]:.1f}s"
                )
                ax.set_ylabel("Heart Rate [bpm]")
                ax.grid(True)

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()
    logger.info(f"Segmented hr plots saved to: {pdf_path}")


def report_unified_hr_baseline(
    shared_reference,
    output_dir,
    report_name="10_hr_baseline_unified_REPORT.pdf",
    title=None,
):
    """Render a unified HR baseline overview panel (HR + actogram + histogram)."""
    from fmcg.analysis.actography import _plot_actography
    from fmcg.analysis.hr import _plot_hr_histogram, _plot_hr_with_segments

    heart_rate = np.asarray(shared_reference.get("heart_rate", []))
    if heart_rate.size == 0:
        return

    segments = shared_reference.get("segments", [])
    baseline_hr = shared_reference.get("baseline_hr")
    bpm_threshold = shared_reference.get("bpm_threshold")

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[2, 1], height_ratios=[2, 1], hspace=0.3, wspace=0.25
    )

    source = shared_reference.get("source")
    chart_title = title or (
        f"Shared-Reference Heart Rate ({source})"
        if source
        else "Shared-Reference Heart Rate"
    )

    ax_hr = fig.add_subplot(gs[0, 0])
    _plot_hr_with_segments(
        heart_rate,
        segments,
        baseline_hr=baseline_hr,
        bpm_threshold=bpm_threshold,
        title=chart_title,
        ax=ax_hr,
    )

    metrics_text = (
        f"Mean HR: {np.nanmean(heart_rate):.2f} bpm\n"
        f"Std HR: {np.nanstd(heart_rate):.2f} bpm\n"
        f"Intervals: {len(heart_rate)}\n"
        f"fs: {float(shared_reference.get('fs', np.nan)):.2f} Hz"
    )
    ax_hr.text(
        0.01,
        0.98,
        metrics_text,
        transform=ax_hr.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax_acto = fig.add_subplot(gs[1, 0], sharex=ax_hr)
    vss = shared_reference.get("vss")
    time_axis = shared_reference.get("time_axis")
    detections = shared_reference.get("detections")
    if vss is not None and time_axis is not None and detections is not None:
        _plot_actography(
            np.asarray(time_axis),
            np.asarray(vss),
            detection=np.asarray(detections),
            title="Actogram",
            ylabel="Velocity",
            ax=ax_acto,
        )
    else:
        ax_acto.text(0.5, 0.5, "Actogram unavailable", ha="center", va="center")
        ax_acto.set_xlabel("Time [s]")
        ax_acto.set_ylabel("Velocity")
        ax_acto.grid(True, alpha=0.3)

    ax_hist = fig.add_subplot(gs[:, 1])
    _plot_hr_histogram(
        heart_rate,
        title="Heart Rate Distribution",
        xlabel="Heart Rate [BPM]",
        ax=ax_hist,
        baseline_hr=baseline_hr,
        bpm_threshold=bpm_threshold,
    )

    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.94, bottom=0.08, wspace=0.25, hspace=0.3
    )
    save_path = os.path.join(output_dir, report_name)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Unified HR baseline report saved to: {save_path}")


def report_m_hat_segments(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    label=["Fetal", "Maternal"],
    report_name="2_dipole_moments_segments_REPORT.pdf",
):
    """
    Generates segmented plots of dipole moments (`m_hat`) for each entry in `data_dict` and saves them to a multi-page PDF report.

    Each plot displays the x, y, and z components of the dipole moments over time, segmented into windows of specified duration. Artifact regions, if present, are highlighted. The report is formatted for A4 pages, with multiple segments per page.

    Args:
        data_dict (dict): Dictionary containing dipole moment data for each key. Each entry should have a "dipole_moments" ndarray of shape (n_samples, 3), and optionally an "artifacts" boolean array of shape (n_samples,).
        time_ (np.ndarray): 1D array of time points corresponding to the dipole moment samples.
        output_dir (str): Directory path where the PDF report will be saved.
        segment_duration (int, optional): Duration (in seconds) of each segment to plot. Defaults to 5.
        label (list of str, optional): List of labels for each key in `data_dict`, used in plot titles. Defaults to ["Fetal", "Maternal"].
    """

    # Create PDF backend
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    # Constants for A4 size and layout
    A4_WIDTH, A4_HEIGHT = 8.27, 11.69  # inches
    ROW_HEIGHT = 2  # inches per row
    MAX_ROWS = int((A4_HEIGHT - 1) / ROW_HEIGHT)  # Leave 1 inch for title and margins

    # Calculate segment indices
    fs = int(1 / (time_[1] - time_[0]))  # Sampling frequency
    segment_samples = segment_duration * fs
    num_segments = int(np.ceil(len(time_) / segment_samples))

    for k, key in enumerate(data_dict.keys()):
        # skip if no dipole moments are available
        if "dipole_moments" not in data_dict[key]:
            continue

        m_hat = data_dict[key]["dipole_moments"]

        for page_idx in range(0, num_segments, MAX_ROWS):
            fig, axes = _create_segment_page(MAX_ROWS, A4_WIDTH, A4_HEIGHT)

            # disable axes that are not used
            _disable_unused_axes(axes, page_idx, num_segments)

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], m_hat[start:end, 0], label="x", color="C0")
                ax.plot(time_[start:end], m_hat[start:end, 1], label="y", color="C1")
                ax.plot(time_[start:end], m_hat[start:end, 2], label="z", color="C2")
                ax.set_title(
                    f"{label[k]}: {time_[start]:.1f}s to {time_[end - 1]:.1f}s"
                )
                ax.set_ylabel("Magnetic Moment [$\mathrm{\mu}$Am$^2$]")
                ax.grid(True)
                ax.minorticks_on()
                ax.grid(which="minor", linestyle=":", alpha=0.2)
                ax.legend(loc="upper right")

                if (
                    "artifacts" in data_dict[key]
                    and len(data_dict[key]["artifacts"]) > 0
                ):
                    ax.fill_between(
                        time_[start:end],
                        np.nanmin(data_dict[key]["dipole_moments"]),
                        np.nanmax(data_dict[key]["dipole_moments"]),
                        where=data_dict[key]["artifacts"][start:end],
                        alpha=0.25,
                        zorder=0,
                        interpolate=False,
                        label="Artifact",
                        color="orange",
                    )

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

    logger.info(f"Segmented m_hat plots saved to: {pdf_path}")


def report_m_hat_segments_per_channel(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    start_seconds=None,
    end_seconds=None,
    report_name="2_dipole_moments_channels_segments_REPORT.pdf",
    max_rows_per_page=6,
    row_hspace=0.15,
):
    """Generate segmented m_hat reports with one row per dipole component channel."""
    if len(time_) < 2 or not data_dict:
        return

    os.makedirs(output_dir, exist_ok=True)

    dt = float(np.median(np.diff(time_)))
    if not np.isfinite(dt) or dt <= 0:
        return
    fs = int(round(1.0 / dt))

    start_idx = (
        0
        if start_seconds is None
        else int(np.searchsorted(time_, start_seconds, side="left"))
    )
    end_idx = (
        len(time_)
        if end_seconds is None
        else int(np.searchsorted(time_, end_seconds, side="right"))
    )
    start_idx = max(0, min(start_idx, len(time_) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(time_)))

    segment_samples = max(1, int(round(segment_duration * fs)))
    n_total = end_idx - start_idx
    n_segments = int(np.ceil(n_total / segment_samples))

    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    max_rows = max(3, int(max_rows_per_page))
    channels_per_group = 3
    groups_per_page = max(1, max_rows // channels_per_group)

    for key, value in data_dict.items():
        m_hat = np.asarray(value.get("dipole_moments"))
        if m_hat.ndim != 2:
            continue

        m_hat_window = m_hat[start_idx:end_idx]
        valid_channels = np.where(np.any(np.isfinite(m_hat_window), axis=0))[0]
        if valid_channels.size == 0:
            continue

        artifacts = value.get("artifacts")

        # Build grouped jobs: each job is one segment with a block of up to x/y/z channels.
        group_jobs = []
        for segment_idx in range(n_segments):
            seg_start = start_idx + segment_idx * segment_samples
            seg_end = min(start_idx + (segment_idx + 1) * segment_samples, end_idx)
            # Skip degenerate segments that would create empty-looking panels/pages.
            if seg_end - seg_start < 2:
                continue
            for ch_start in range(0, valid_channels.size, channels_per_group):
                channel_block = valid_channels[ch_start : ch_start + channels_per_group]
                if channel_block.size == 0:
                    continue
                seg_block = m_hat[seg_start:seg_end][:, channel_block]
                if not np.any(np.isfinite(seg_block)):
                    continue
                group_jobs.append((seg_start, seg_end, channel_block))

        if not group_jobs:
            continue

        for page_start in range(0, len(group_jobs), groups_per_page):
            page_groups = group_jobs[page_start : page_start + groups_per_page]
            if not page_groups:
                continue

            page_nrows = channels_per_group * len(page_groups)

            fig, axes = plt.subplots(
                nrows=page_nrows,
                ncols=1,
                figsize=(A4_WIDTH, A4_HEIGHT),
                sharex=False,
                dpi=300,
            )
            axes = np.atleast_1d(axes)

            # Hide all axes first; only used slots are turned on below.
            for ax in axes:
                ax.axis("off")

            plotted_any = False

            for panel_idx, (seg_start, seg_end, channel_block) in enumerate(
                page_groups
            ):
                row_base = panel_idx * channels_per_group
                seg_time = time_[seg_start:seg_end]
                if seg_time.size < 2:
                    continue

                if seg_time.size > 1 and float(seg_time[-1]) > float(seg_time[0]):
                    xlim_left = float(seg_time[0])
                    xlim_right = float(seg_time[-1])
                else:
                    # Fallback for degenerate single-sample windows.
                    center = float(seg_time[0])
                    half_width = 0.5 / max(fs, 1)
                    xlim_left = center - half_width
                    xlim_right = center + half_width

                for local_row in range(channels_per_group):
                    ax = axes[row_base + local_row]
                    ax.axis("on")
                    if local_row >= channel_block.size:
                        ax.axis("off")
                        continue

                    channel_idx = int(channel_block[local_row])
                    trace = m_hat[seg_start:seg_end, channel_idx]
                    if not np.any(np.isfinite(trace)):
                        ax.axis("off")
                        continue

                    plotted_any = True
                    comp_label = (
                        f"M_{'xyz'[channel_idx]}"
                        if channel_idx < 3
                        else f"M_{channel_idx}"
                    )
                    comp_color = f"C{channel_idx % 3}"
                    ax.plot(seg_time, trace, color=comp_color, linewidth=0.8)

                    if artifacts is not None and len(artifacts) == len(time_):
                        y_min = np.nanmin(trace) if np.any(np.isfinite(trace)) else -1.0
                        y_max = np.nanmax(trace) if np.any(np.isfinite(trace)) else 1.0
                        ax.fill_between(
                            seg_time,
                            y_min,
                            y_max,
                            where=np.asarray(artifacts[seg_start:seg_end]).astype(bool),
                            alpha=0.12,
                            zorder=0,
                            interpolate=False,
                            color="orange",
                        )

                    ax.set_ylabel(comp_label, fontsize=7, rotation=0, labelpad=12)
                    ax.tick_params(axis="y", labelsize=6)
                    ax.grid(True, alpha=0.25)
                    ax.minorticks_on()
                    ax.grid(which="minor", linestyle=":", alpha=0.15)
                    ax.set_xlim(xlim_left, xlim_right)

                    if local_row == 0:
                        ch_lo = int(channel_block[0])
                        ch_hi = int(channel_block[-1])
                        ax.set_title(
                            f"{key} | m_hat ch {ch_lo}-{ch_hi} | {seg_time[0]:.1f}s to {seg_time[-1]:.1f}s",
                            fontsize=7,
                            pad=1.5,
                        )
                    if local_row < channels_per_group - 1:
                        ax.tick_params(axis="x", labelbottom=False)
                    else:
                        ax.set_xlabel("Time [s]")

            page_end = min(page_start + len(page_groups), len(group_jobs))
            fig.suptitle(
                f"{key} | m_hat per-channel groups {page_start + 1}-{page_end}",
                y=0.995,
                fontsize=9,
            )
            fig.subplots_adjust(
                left=0.09,
                right=0.995,
                top=0.97,
                bottom=0.045,
                hspace=row_hspace,
            )
            if plotted_any:
                pdf.savefig(fig)
            plt.close(fig)

    pdf.close()
    logger.info(f"Segmented m_hat channel report saved to: {pdf_path}")


def report_r_hat_segments(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    label=["Fetal", "Maternal"],
    window_boundaries=None,
    report_name="2_dipole_positions_segments_REPORT.pdf",
):
    """
    Generates segmented plots of dipole positions (`r_hat`) for each entry in `data_dict` and saves them to a multi-page PDF report.

    Each plot displays the x, y, and z components of the dipole positions over time, segmented into windows of specified duration. Artifact regions, if present, are highlighted. The report is formatted for A4 pages, with multiple segments per page.

    Args:
        data_dict (dict): Dictionary containing dipole position data for each key. Each entry should have a "position" ndarray of shape (n_samples, 3), and optionally an "artifacts" boolean array of shape (n_samples,).
        time_ (np.ndarray): 1D array of time points corresponding to the dipole position samples.
        output_dir (str): Directory path where the PDF report will be saved.
        segment_duration (int, optional): Duration (in seconds) of each segment to plot. Defaults to 5.
        label (list of str, optional): List of labels for each key in `data_dict`, used in plot titles. Defaults to ["Fetal", "Maternal"].
        window_boundaries (list of tuple, optional): List of (start_idx, end_idx) for processing windows to visualize.
    """

    # Create PDF backend
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    # Constants for A4 size and layout
    A4_WIDTH, A4_HEIGHT = 8.27, 11.69  # inches
    ROW_HEIGHT = 2  # inches per row
    MAX_ROWS = int((A4_HEIGHT - 1) / ROW_HEIGHT)  # Leave 1 inch for title and margins

    # Calculate segment indices
    fs = int(1 / (time_[1] - time_[0]))  # Sampling frequency
    segment_samples = segment_duration * fs
    num_segments = int(np.ceil(len(time_) / segment_samples))

    for k, key in enumerate(data_dict.keys()):
        # skip if no dipole positions are available
        if "position" not in data_dict[key]:
            continue

        r_hat = data_dict[key]["position"] * 100  # Convert to cm for display

        for page_idx in range(0, num_segments, MAX_ROWS):
            fig, axes = _create_segment_page(MAX_ROWS, A4_WIDTH, A4_HEIGHT)

            # disable axes that are not used
            _disable_unused_axes(axes, page_idx, num_segments)

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], r_hat[start:end, 0], label="x", zorder=3)
                ax.plot(time_[start:end], r_hat[start:end, 1], label="y", zorder=3)
                ax.plot(time_[start:end], r_hat[start:end, 2], label="z", zorder=3)
                ax.set_title(
                    f"{label[k]}: {time_[start]:.1f}s to {time_[end - 1]:.1f}s"
                )
                ax.set_ylabel("Position [cm]")
                ax.grid(True, zorder=0)
                ax.legend(loc="upper right")

                # Highlight artifact regions
                if (
                    "artifacts" in data_dict[key]
                    and len(data_dict[key]["artifacts"]) > 0
                ):
                    # Get y-limits for the current segment
                    segment_data = r_hat[start:end]
                    y_min = (
                        np.nanmin(segment_data)
                        if not np.all(np.isnan(segment_data))
                        else -1
                    )
                    y_max = (
                        np.nanmax(segment_data)
                        if not np.all(np.isnan(segment_data))
                        else 1
                    )

                    ax.fill_between(
                        time_[start:end],
                        y_min,
                        y_max,
                        where=data_dict[key]["artifacts"][start:end],
                        alpha=0.3,
                        zorder=1,
                        interpolate=False,
                        label="Artifact",
                        color="red",
                    )

                # Plot window boundaries if provided
                if window_boundaries:
                    plotted_boundaries = False
                    for w_start, w_end in window_boundaries:
                        # Convert indices to time
                        if w_start < len(time_):
                            t_start = time_[w_start]
                            # Check if start is within current plot range
                            if t_start >= time_[start] and t_start <= time_[end - 1]:
                                ax.axvline(
                                    x=t_start,
                                    color="blue",
                                    linestyle="--",
                                    alpha=0.6,
                                    linewidth=1.5,
                                    zorder=2,
                                    label=(
                                        "Window Start" if not plotted_boundaries else ""
                                    ),
                                )
                                plotted_boundaries = True

                        # Handle end index
                        if w_end < len(time_):
                            t_end = time_[w_end]
                        else:
                            t_end = time_[-1]

                        # Check if end is within current plot range
                        if t_end >= time_[start] and t_end <= time_[end - 1]:
                            ax.axvline(
                                x=t_end,
                                color="green",
                                linestyle=":",
                                alpha=0.6,
                                linewidth=1.5,
                                zorder=2,
                                label="Window End" if not plotted_boundaries else "",
                            )

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

    logger.info(f"Segmented r_hat plots saved to: {pdf_path}")


def report_ica_components(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
):
    """
    Generates a segmented PDF report visualizing ICA components, detected peaks, outliers, and artifacts.

    The function divides the time series data into segments and plots each segment of the ICA components.
    Peaks and outliers are highlighted, and artifact regions are shaded if provided. Each segment is plotted
    on a separate row, and multiple segments are grouped per PDF page according to A4 size constraints.

    Args:
        data_dict (dict): Dictionary containing ICA component data for each key. Each entry should have:
            - "components" (np.ndarray): ICA component time series.
            - "peaks" (np.ndarray): Indices of detected peaks.
            - "outlier" (np.ndarray): Boolean array indicating outlier peaks.
            - "artifacts" (optional, np.ndarray): Boolean array indicating artifact regions.
        time_ (np.ndarray): Array of time points corresponding to the data samples.
        output_dir (str): Directory path where the PDF report will be saved.
        segment_duration (int, optional): Duration (in seconds) of each segment to plot per row. Default is 5.
    """

    # Calculate sampling frequency from time array
    if len(time_) > 1:
        fs = int(1 / (time_[1] - time_[0]))  # Sampling frequency
    else:
        fs = 1000  # Default fallback

    # Create PDF backend
    pdf_path = os.path.join(output_dir, "3_separated_ica-components_REPORT.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    # Constants for A4 size and layout
    A4_WIDTH, A4_HEIGHT = 8.27, 11.69  # inches
    ROW_HEIGHT = 2  # inches per row
    MAX_ROWS = int((A4_HEIGHT - 1) / ROW_HEIGHT)  # Leave 1 inch for title and margins

    # Calculate segment indices
    segment_samples = segment_duration * fs
    num_segments = int(np.ceil(len(time_) / segment_samples))

    for key in data_dict.keys():
        component = data_dict[key]["components"].copy()

        for page_idx in range(0, num_segments, MAX_ROWS):
            fig, axes = _create_segment_page(MAX_ROWS, A4_WIDTH, A4_HEIGHT)

            # disable axes that are not used
            _disable_unused_axes(axes, page_idx, num_segments)

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], component[start:end])
                peaks = data_dict[key]["peaks"]
                ax.plot(
                    time_[peaks[(peaks >= start) & (peaks < end)]],
                    component[peaks[(peaks >= start) & (peaks < end)]],
                    "rx",
                )

                _plot_segment_outliers(
                    ax,
                    time_,
                    component,
                    peaks,
                    data_dict[key]["outlier"],
                    start,
                    end,
                )

                if "artifacts" in data_dict[key]:
                    ax.fill_between(
                        time_[start:end],
                        np.nanmin(component),
                        np.nanmax(component),
                        where=data_dict[key]["artifacts"][start:end],
                        alpha=0.25,
                        zorder=0,
                        interpolate=False,
                        label="Artifact",
                        color="orange",
                    )

                ax.legend(loc="upper right")
                ax.set_title(
                    f"{key.capitalize()}: {time_[start]:.1f}s to {time_[end - 1]:.1f}s"
                )
                ax.set_ylabel("Separated ICA-Component")
                ax.grid(True)

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

    logger.info(f"Segmented hr plots saved to: {pdf_path}")


def report_field_segments(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    report_name="4_field_segments_REPORT.pdf",
    max_rows_per_page=6,
    row_hspace=0.25,
):
    """Generate A4 multi-page segmented field reports with Holter-like traces."""
    if len(time_) < 2 or not data_dict:
        return

    os.makedirs(output_dir, exist_ok=True)

    fs = int(1 / (time_[1] - time_[0]))
    segment_samples = max(1, int(segment_duration * fs))
    num_segments = int(np.ceil(len(time_) / segment_samples))

    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    max_rows = max(1, int(max_rows_per_page))

    for key, value in data_dict.items():
        field = np.asarray(value.get("field"))
        if field.ndim == 3:
            field = field.reshape(field.shape[0], -1)
        if field.ndim != 2:
            continue

        selected_spans = value.get("selected_spans", [])

        for page_idx in range(0, num_segments, max_rows):
            nrows = min(max_rows, num_segments - page_idx)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                figsize=(A4_WIDTH, A4_HEIGHT),
                sharex=False,
            )
            axes = np.atleast_1d(axes)

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                seg_time = time_[start:end]
                seg_field = field[start:end]
                ax.plot(seg_time, seg_field, color="k", alpha=0.30, linewidth=0.5)
                finite = np.isfinite(seg_field)
                finite_count = np.sum(finite, axis=1)

                magnitude = np.sqrt(np.nansum(np.square(seg_field), axis=1))
                magnitude[finite_count == 0] = np.nan
                ax.plot(
                    seg_time,
                    magnitude,
                    color="k",
                    linewidth=0.5,
                    label="Magnitude",
                    linestyle="--",
                    alpha=0.30,
                )

                for span_idx, (t0, t1) in enumerate(selected_spans):
                    if t1 < seg_time[0] or t0 > seg_time[-1]:
                        continue
                    label = "Selected range" if span_idx == 0 and row_idx == 0 else None
                    ax.axvspan(
                        max(t0, seg_time[0]),
                        min(t1, seg_time[-1]),
                        color="tab:grey",
                        alpha=0.10,
                        label=label,
                    )

                ax.set_title(f"{key}: {seg_time[0]:.1f}s to {seg_time[-1]:.1f}s")
                ax.set_ylabel("Field [a.u.]")
                ax.grid(True, alpha=0.3)
                ax.minorticks_on()
                ax.grid(which="minor", linestyle=":", alpha=0.2)
                if row_idx == 0:
                    ax.legend(loc="upper right", fontsize=7)

            axes[-1].set_xlabel("Time [s]")
            fig.subplots_adjust(
                left=0.07,
                right=0.995,
                top=0.985,
                bottom=0.045,
                hspace=row_hspace,
            )
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()
    logger.info(f"Segmented field plots saved to: {pdf_path}")


def report_field_segments_per_channel(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    start_seconds=None,
    end_seconds=None,
    report_name="4_field_channels_segments_REPORT.pdf",
    max_rows_per_page=16,
    row_hspace=0.03,
):
    """Generate one-page-per-segment field reports with one row per valid channel."""
    if len(time_) < 2 or not data_dict:
        return

    os.makedirs(output_dir, exist_ok=True)

    dt = float(np.median(np.diff(time_)))
    if not np.isfinite(dt) or dt <= 0:
        return
    fs = int(round(1.0 / dt))

    start_idx = (
        0
        if start_seconds is None
        else int(np.searchsorted(time_, start_seconds, side="left"))
    )
    end_idx = (
        len(time_)
        if end_seconds is None
        else int(np.searchsorted(time_, end_seconds, side="right"))
    )
    start_idx = max(0, min(start_idx, len(time_) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(time_)))

    segment_samples = max(1, int(round(segment_duration * fs)))
    n_total = end_idx - start_idx
    n_segments = int(np.ceil(n_total / segment_samples))

    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    max_rows = max(1, int(max_rows_per_page))

    for key, value in data_dict.items():
        field = np.asarray(value.get("field"))
        if field.ndim == 3:
            field = field.reshape(field.shape[0], -1)
        if field.ndim != 2:
            continue

        field_window = field[start_idx:end_idx]
        valid_channels = np.where(np.any(np.isfinite(field_window), axis=0))[0]
        if valid_channels.size == 0:
            continue

        selected_spans = value.get("selected_spans", [])

        for segment_idx in range(n_segments):
            seg_start = start_idx + segment_idx * segment_samples
            seg_end = min(start_idx + (segment_idx + 1) * segment_samples, end_idx)
            seg_time = time_[seg_start:seg_end]
            if seg_time.size == 0:
                continue

            for ch_start in range(0, valid_channels.size, max_rows):
                channel_block = valid_channels[ch_start : ch_start + max_rows]
                nrows = int(channel_block.size)
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=1,
                    figsize=(A4_WIDTH, A4_HEIGHT),
                    sharex=True,
                    dpi=300,
                )
                axes = np.atleast_1d(axes)

                seg_field = field[seg_start:seg_end][:, channel_block]

                for row_idx, ax in enumerate(axes):
                    channel_idx = int(channel_block[row_idx])
                    trace = seg_field[:, row_idx]
                    ax.plot(seg_time, trace, color="k", linewidth=0.45)

                    for t0, t1 in selected_spans:
                        if t1 < seg_time[0] or t0 > seg_time[-1]:
                            continue
                        ax.axvspan(
                            max(t0, seg_time[0]),
                            min(t1, seg_time[-1]),
                            color="tab:grey",
                            alpha=0.08,
                        )

                    ax.set_ylabel(
                        f"ch{channel_idx}", fontsize=6, rotation=0, labelpad=14
                    )
                    ax.tick_params(axis="y", labelsize=6)
                    ax.grid(True, alpha=0.2)
                    ax.minorticks_on()
                    ax.grid(which="minor", linestyle=":", alpha=0.15)

                ch_lo = int(channel_block[0])
                ch_hi = int(channel_block[-1])
                axes[0].set_title(
                    f"{key} | channels {ch_lo}-{ch_hi} | {seg_time[0]:.1f}s to {seg_time[-1]:.1f}s",
                    fontsize=8,
                    pad=2,
                )
                axes[-1].set_xlabel("Time [s]")
                fig.subplots_adjust(
                    left=0.09,
                    right=0.995,
                    top=0.985,
                    bottom=0.045,
                    hspace=row_hspace,
                )
                pdf.savefig(fig)
                plt.close(fig)

    pdf.close()
    logger.info(f"Segmented field channel report saved to: {pdf_path}")


def report_topk_averages(
    method_name,
    averages,
    output_dir,
    report_name=None,
    ncol=2,
    top_k=3,
):
    """Generate an A4 top-k averaged heartbeat report for a single method."""
    from fmcg.analysis.heartbeat_averaging import _plot_average_heartbeat

    ncol = max(1, int(ncol))
    top_k = max(1, int(top_k))

    entries = list(averages[:top_k]) if averages else []
    if not entries:
        return

    if report_name is None:
        report_name = f"{method_name}_topk_averages_REPORT.pdf"

    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    panel_height = 2.5
    nrows = max(1, int((A4_HEIGHT - 0.9) / panel_height))
    panels_per_page = nrows * max(1, ncol)

    for page_start in range(0, len(entries), panels_per_page):
        page_entries = entries[page_start : page_start + panels_per_page]
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncol,
            figsize=(A4_WIDTH, A4_HEIGHT),
            dpi=300,
        )
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, (rank, avg) in zip(axes, enumerate(page_entries, start=page_start + 1)):
            ax.axis("on")
            title = f"{method_name}"
            seg_details = _format_segment_details(avg.get("segment"))
            if seg_details:
                title = f"{title} | {seg_details}"

            _plot_average_heartbeat(
                np.asarray(avg["mean"]),
                np.asarray(avg["std"]),
                np.asarray(avg["time_axis"]),
                title=title,
                ax=ax,
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    logger.info(f"Top-k averaged beat report ({method_name}) saved to: {pdf_path}")


# ------------------------ Comparison reports ------------------------


def report_topk_average_comparison(
    topk_data,
    output_dir,
    report_name="6_average_comparison_by_rank_REPORT.pdf",
    ncol=2,
    top_k=3,
):
    """Generate A4 comparison report: one rank k per page set, all approaches on that rank."""
    from fmcg.analysis.heartbeat_averaging import _plot_average_heartbeat

    ncol = max(1, int(ncol))
    top_k = max(1, int(top_k))

    if not topk_data:
        return

    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, report_name)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    A4_WIDTH, A4_HEIGHT = 8.27, 11.69
    panel_height = 2.6
    nrows = max(1, int((A4_HEIGHT - 1.0) / panel_height))
    panels_per_page = nrows * max(1, ncol)

    for rank_idx in range(top_k):
        rank_entries = []
        for method_name, avgs in topk_data.items():
            if avgs and len(avgs) > rank_idx:
                rank_entries.append((method_name, avgs[rank_idx]))

        if not rank_entries:
            continue

        # Keep pages grouped by the same rank before moving to next rank.
        for page_start in range(0, len(rank_entries), panels_per_page):
            page_entries = rank_entries[page_start : page_start + panels_per_page]
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncol,
                figsize=(A4_WIDTH, A4_HEIGHT),
                dpi=300,
            )
            axes = np.array(axes).reshape(-1)

            for ax in axes:
                ax.axis("off")

            for ax, (method_name, avg) in zip(axes, page_entries):
                ax.axis("on")
                title = f"{method_name}"
                seg_details = _format_segment_details(avg.get("segment"))
                if seg_details:
                    title = f"{title} | {seg_details}"

                _plot_average_heartbeat(
                    np.asarray(avg["mean"]),
                    np.asarray(avg["std"]),
                    np.asarray(avg["time_axis"]),
                    title=title,
                    ax=ax,
                )

            fig.suptitle(
                f"Average Comparison Across Methods - Segment {rank_idx + 1}", y=0.995
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()
    logger.info(f"Average comparison-by-rank report saved to: {pdf_path}")


def report_excerpt_comparison(
    method_signals,
    fs,
    output_dir,
    report_name="excerpt_comparison_REPORT.pdf",
    start_seconds=30.0,
    seconds=5.0,
    selected_spans=None,
    title="Method Comparison: Unaveraged Field Excerpts",
):
    """Render a multi-method field excerpt comparison with optional span overlays."""
    if not method_signals or len(method_signals) < 2:
        return

    names = []
    signals = []
    for name, signal in method_signals.items():
        arr = np.asarray(signal)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        names.append(name)
        signals.append(arr)

    if len(signals) < 2:
        return

    fs = float(fs)
    n = max(1, int(seconds * fs))
    max_start = max(0, min(signal.shape[0] for signal in signals) - n)
    start_idx = int(round(start_seconds * fs))
    start_idx = max(0, min(start_idx, max_start))
    end_idx = start_idx + n
    time_excerpt = np.arange(start_idx, end_idx) / fs

    os.makedirs(output_dir, exist_ok=True)
    n_methods = len(signals)
    fig, axes = plt.subplots(
        n_methods, 1, figsize=(12, max(2.2 * n_methods, 3.5)), sharex=True, dpi=300
    )
    if n_methods == 1:
        axes = [axes]

    spans = list(selected_spans or [])
    for idx, (ax, method_name, signal_2d) in enumerate(zip(axes, names, signals)):
        excerpt = signal_2d[start_idx:end_idx]
        ax.plot(time_excerpt, excerpt, color="k", alpha=0.3, linewidth=0.22)
        ax.plot(time_excerpt, np.linalg.norm(excerpt, axis=1), color="r", linewidth=0.7)
        for span_idx, (t0, t1) in enumerate(spans):
            if t1 < time_excerpt[0] or t0 > time_excerpt[-1]:
                continue
            lbl = "Selected baseline range" if idx == 0 and span_idx == 0 else None
            ax.axvspan(
                max(t0, float(time_excerpt[0])),
                min(t1, float(time_excerpt[-1])),
                color="tab:grey",
                alpha=0.10,
                label=lbl,
            )
        ax.set_ylabel(method_name)
        ax.grid(True, alpha=0.3)
        if idx == 0 and spans:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(float(time_excerpt[0]), float(time_excerpt[-1]))
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    save_path = os.path.join(output_dir, report_name)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Excerpt comparison report saved to: {save_path}")
