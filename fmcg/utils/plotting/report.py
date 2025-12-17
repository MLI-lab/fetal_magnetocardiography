import os

import matplotlib.backends.backend_pdf
import numpy as np
from matplotlib import pyplot as plt

from fmcg.utils.plotting.plot_utils import logger


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
            fig, axes = plt.subplots(
                nrows=MAX_ROWS, ncols=1, figsize=(A4_WIDTH, A4_HEIGHT), sharex=False
            )

            # disable axes that are not used
            for i in range(MAX_ROWS):
                if page_idx + i >= num_segments:
                    axes[i].axis("off")

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], hr[start:end])

                # Handle peaks and outliers differently for segmented vs non-segmented processing
                peaks = data_dict[key]["peaks"]

                # Check if we have segmented processing (artifacts indicates full timeline reconstruction)
                if "artifacts" in data_dict[key] and len(
                    data_dict[key]["outlier"]
                ) == len(time_):
                    # For segmented processing with full timeline reconstruction
                    peaks_in_segment = peaks[(peaks >= start) & (peaks < end)]
                    if len(peaks_in_segment) > 0:
                        # Use corresponding outlier mask for the segment
                        outlier_for_peaks = data_dict[key]["outlier"][peaks_in_segment]
                        ax.fill_between(
                            time_[peaks_in_segment],
                            np.nanmin(hr[start:end]),
                            np.nanmax(hr[start:end]),
                            where=outlier_for_peaks,
                            alpha=0.25,
                            zorder=0,
                            interpolate=False,
                            label="Outlier",
                            color="grey",
                        )
                else:
                    # Original behavior for non-segmented processing
                    peaks_in_segment = peaks[(peaks >= start) & (peaks < end)]
                    if len(peaks_in_segment) > 0 and len(peaks_in_segment) <= len(
                        data_dict[key]["outlier"]
                    ):
                        # Ensure we don't exceed outlier array bounds
                        outlier_indices = (
                            peaks_in_segment - start
                        )  # Convert to local indices
                        outlier_indices = outlier_indices[
                            outlier_indices < len(data_dict[key]["outlier"])
                        ]
                        if len(outlier_indices) > 0:
                            ax.fill_between(
                                time_[peaks_in_segment[: len(outlier_indices)]],
                                np.nanmin(hr[start:end]),
                                np.nanmax(hr[start:end]),
                                where=data_dict[key]["outlier"][outlier_indices],
                                alpha=0.25,
                                zorder=0,
                                interpolate=False,
                                label="Outlier",
                                color="grey",
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


def report_m_hat_segments(
    data_dict,
    time_,
    output_dir,
    segment_duration=5,
    label=["Fetal", "Maternal"],
    window_boundaries=None,
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
        window_boundaries (list of tuple, optional): List of (start_idx, end_idx) for processing windows to visualize.
    """

    # Create PDF backend
    pdf_path = os.path.join(output_dir, "2_dipole_moments_REPORT.pdf")
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
            fig, axes = plt.subplots(
                nrows=MAX_ROWS, ncols=1, figsize=(A4_WIDTH, A4_HEIGHT), sharex=False
            )

            # disable axes that are not used
            for i in range(MAX_ROWS):
                if page_idx + i >= num_segments:
                    axes[i].axis("off")

            for row_idx, ax in enumerate(axes):
                segment_idx = page_idx + row_idx
                if segment_idx >= num_segments:
                    break

                start = segment_idx * segment_samples
                end = min((segment_idx + 1) * segment_samples, len(time_))

                ax.plot(time_[start:end], m_hat[start:end, 0], label="x")
                ax.plot(time_[start:end], m_hat[start:end, 1], label="y")
                ax.plot(time_[start:end], m_hat[start:end, 2], label="z")
                ax.set_title(
                    f"{label[k]}: {time_[start]:.1f}s to {time_[end - 1]:.1f}s"
                )
                ax.set_ylabel("Magnetic Moment [$\mathrm{\mu}$Am$^2$]")
                ax.grid(True)
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
                
                # Plot window boundaries if provided
                if window_boundaries:
                    for w_start, w_end in window_boundaries:
                        # Convert indices to time
                        if w_start < len(time_):
                            t_start = time_[w_start]
                            # Check if start is within current plot range
                            if t_start >= time_[start] and t_start <= time_[end - 1]:
                                ax.axvline(x=t_start, color='g', linestyle='--', alpha=0.5, linewidth=1)
                        
                        # Handle end index
                        if w_end < len(time_):
                            t_end = time_[w_end]
                        else:
                            t_end = time_[-1]

                        # Check if end is within current plot range
                        if t_end >= time_[start] and t_end <= time_[end - 1]:
                            ax.axvline(x=t_end, color='r', linestyle=':', alpha=0.5, linewidth=1)

            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf.close()

    logger.info(f"Segmented m_hat plots saved to: {pdf_path}")


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
    fs = int(1 / (time_[1] - time_[0]))  # Sampling frequency
    segment_samples = segment_duration * fs
    num_segments = int(np.ceil(len(time_) / segment_samples))

    for key in data_dict.keys():
        component = data_dict[key]["components"].copy()

        for page_idx in range(0, num_segments, MAX_ROWS):
            fig, axes = plt.subplots(
                nrows=MAX_ROWS, ncols=1, figsize=(A4_WIDTH, A4_HEIGHT), sharex=False
            )

            # disable axes that are not used
            for i in range(MAX_ROWS):
                if page_idx + i >= num_segments:
                    axes[i].axis("off")

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

                peaks_in_segment = peaks[(peaks >= start) & (peaks < end)]
                if len(peaks_in_segment) > 0 and len(peaks_in_segment) <= len(
                    data_dict[key]["outlier"]
                ):
                    # Ensure we don't exceed outlier array bounds
                    outlier_indices = (
                        peaks_in_segment - start
                    )  # Convert to local indices
                    outlier_indices = outlier_indices[
                        outlier_indices < len(data_dict[key]["outlier"])
                    ]
                    if len(outlier_indices) > 0:
                        ax.fill_between(
                            time_[peaks_in_segment[: len(outlier_indices)]],
                            np.nanmin(component[start:end]),
                            np.nanmax(component[start:end]),
                            where=data_dict[key]["outlier"][outlier_indices],
                            alpha=0.25,
                            zorder=0,
                            interpolate=False,
                            label="Outlier",
                            color="grey",
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