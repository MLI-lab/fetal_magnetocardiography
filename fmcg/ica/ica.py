import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

from ..utils.utils import reduced2full
from ._cluster_components import _cluster_components
from ..analysis.hr import detect_hr_outlier

def apply_ICA(signal, n_comp=None, seed=42):
    """Applies FastICA to the input signal."""
    ica = FastICA(n_components=n_comp, max_iter=500, tol=1e-5, random_state=seed)
    S_ = ica.fit_transform(signal)
    return S_, ica


def compute_ICA_statistics(sources, fs, offset=10, span=5, mixing_matrix=None, axis_mask=None):
    """
    Computes statistics for all ICA components.
    """
    n_components = sources.shape[1]
    stats_list = []

    # 1. Calculate statistics for ALL components
    for j in range(n_components):
        signal = sources[int(offset * fs) : int(offset * fs + span * fs), j]

        # Basic Kurtosis
        kurt = kurtosis(signal)

        # Spatial energy and dispersion (if mixing matrix and axis mask are provided)
        if mixing_matrix is not None:
            energy = np.sum(mixing_matrix[:, j] ** 2)
        if mixing_matrix is not None and axis_mask is not None:
            field_variance = np.mean([tv_isotropic(rmvnan(weight2map(mixing_matrix[:, j], axis_mask, coord=c))) for c in range(3)])

        # Signal processing for peak detection
        try:
            # Invert if necessary and clean
            signal_inv, _ = nk.ecg_invert(signal, fs)
            signal_clean = nk.ecg_clean(signal_inv, sampling_rate=fs, method="vg")
            peaks_dict = nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")
            peaks = peaks_dict["ECG_R_Peaks"]

            if len(peaks) > 1:
                # Stats computations
                snr = 10 * np.log10(np.mean(signal[peaks] ** 2) / np.mean(signal**2))
                mean_hr = 60 / np.mean(np.diff(peaks) / fs)  # bpm
                sdnn = np.std(np.diff(peaks)) * (1000 / fs)  # ms

                hr_outlier = detect_hr_outlier(peaks, fs)
                hr_outlier_rate = np.sum(hr_outlier) / len(hr_outlier)
            else:
                mean_hr, snr, sdnn, hr_outlier_rate = np.nan, np.nan, 0.0, 0.0
        except:
            # Handle edge cases where neurokit fails
            peaks = []
            mean_hr, snr, sdnn, hr_outlier_rate = np.nan, np.nan, 0.0, 0.0

        stats_list.append(
            {
                "index": j,
                "signal": signal,
                "peaks": peaks,
                "num_peaks": len(peaks),
                "kurtosis": kurt,
                "mean_hr": mean_hr,
                "snr": snr,
                "sdnn": sdnn,
                "hr_outlier_rate": hr_outlier_rate,
            }
        )
        if mixing_matrix is not None:
            stats_list[-1]["energy"] = energy
        if mixing_matrix is not None and axis_mask is not None:
            stats_list[-1]["field_variance"] = field_variance

    # 2. Sort components by SNR (highest SNR first)
    # If you want fetal (usually higher SNR) at the top, keep reverse=True
    stats_list.sort(key=lambda x: x["snr"], reverse=True)

    return stats_list


def select_ica_components(
    sources,
    fs,
    offset=10,
    plot_span=5,
    span=5,
    method="clustering_heuristic",
    selection=None,
    clustering_args=None,
    plot_components=5,
    plot=False,
    mixing_matrix=None,
    axis_mask=None
):
    """ "
    Select ICA components based on heuristics or manual selection.
    Parameters:
        sources (ndarray): The ICA source signals of shape (n_samples, n_components).
        fs (int): Sampling frequency of the signals.
        offset (float): Time in seconds to start analyzing the components. Default is 10 seconds.
        plot_span (float): Time in seconds to plot for each component. Default is 5 seconds.
        span (float): Time in seconds to analyze for each component. Default is 5 seconds.
        method (str): Method for selecting components. Options are "clustering_heuristic" or "manual". Default is "clustering_heuristic".
        selection (tuple): If method is "manual", a tuple of (maternal_idx, fetal_idx) specifying the indices of the maternal and fetal components. Default is None.
        clustering_args (dict): Arguments for clustering when method is "clustering_heuristic". Should include 'n_components' and 'n_clusters'. Default is None.
        plot_components (int): Number of top components to plot for visualization. Default is 5.
        plot (bool): Whether to plot the components and their statistics. Default is False.
        mixing_matrix (ndarray): The ICA mixing matrix of shape (n_channels, n_components). optional for spatial statistics.
        axis_mask (ndarray): Boolean mask of shape (n_channels,) indicating which channels correspond to the fetal heart. optional for spatial statistics.
    Returns:
        maternal (int): Index of the selected maternal component.
        fetal (int): Index of the selected fetal component.
        grouped (dict): Dictionary mapping cluster labels to lists of component indices (only if method is "clustering_heuristic").
    """
    if clustering_args is None:
        clustering_args = {"n_components": 5, "n_clusters": 3}

    # 1. Compute stats
    ica_stats = compute_ICA_statistics(sources, fs, offset=offset, span=span, mixing_matrix=mixing_matrix, axis_mask=axis_mask)

    # 2. Assignment
    if method == "clustering_heuristic":
        maternal, fetal, df_clustered = _cluster_components(
            ica_stats, **clustering_args
        )
        cluster_assignment = dict(
            zip(df_clustered["index"], df_clustered["assigned_label"])
        )
    elif method == "manual":
        if selection is None or len(selection) != 2:
            maternal, fetal = (None, None)
        cluster_assignment = {}
    else:
        raise ValueError(f"Unknown method {method}")

    # 3. Visualization
    if plot:
        stats_list = ica_stats[:plot_components]
        fig, axes = plt.subplots(
            plot_components,
            1,
            sharex=True,
            figsize=(12, 1.5 * plot_components),
            dpi=100,
        )
        if plot_components == 1:
            axes = [axes]

        for i, comp in enumerate(stats_list):
            ax = axes[i]
            sig = comp["signal"][: int(plot_span * fs)]
            pks = comp["peaks"]
            pks = pks[pks < int(plot_span * fs)]
            idx = comp["index"]

            # Determine coloring
            edge_color = "lightgrey"
            title_suffix = ""
            is_selected = False

            if idx == maternal:
                title_suffix = " (Maternal)"
                is_selected = True
                edge_color = "tab:blue"
                line_color = "tab:blue"
            elif idx == fetal:
                title_suffix = " (Fetal)"
                is_selected = True
                edge_color = "tab:red"
                line_color = "tab:red"
            else:
                line_color = "tab:grey"

            # Plot Signal
            time_axis = np.arange(plot_span * fs) / fs
            ax.plot(time_axis, sig, label=f"Comp {idx}", color=line_color, lw=1)

            # Plot Peaks
            if len(pks) > 0:
                ax.plot(pks / fs, sig[pks], "rx", markersize=7, label="Peaks")

            # Annotation Box
            stats_text = (
                f"Energy: {comp.get('energy', np.nan):.2f}\n"
                f"SDNN: {comp['sdnn']:.2f} ms\n"
                f"Kurt: {comp['kurtosis']:.2f}\n"
                f"SNR: {comp['snr']:.2f} dB\n"
                f"Mean HR: {comp['mean_hr']:.2f} BPM\n"
                f"Num Peaks: {len(pks)}\n"
                f"HR Outlier Rate: {comp['hr_outlier_rate']:.2f}\n"
                f"Field Var: {comp.get('field_variance', np.nan):.2f}"
            )

            ax.text(
                1.01,
                0.5,
                stats_text,
                transform=ax.transAxes,
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Title
            title_str = f"ICA Component {idx}"
            if method == "clustering_heuristic" and idx in cluster_assignment:
                title_str += f" - Cluster {cluster_assignment[idx]}"
            title_str += title_suffix

            ax.set_title(
                title_str, fontsize=12, fontweight="bold" if is_selected else "normal"
            )
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Highlight selected components with explicit edging
            if is_selected:
                for spine in ax.spines.values():
                    spine.set_edgecolor(edge_color)
                    spine.set_linewidth(3)

        axes[-1].set_xlabel("Time [s]")
        plt.suptitle("ICA Components Selection", fontsize=14, y=1.02)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show()

    from collections import defaultdict

    grouped = defaultdict(list)
    for k, v in cluster_assignment.items():
        grouped[v].append(k)

    return maternal, fetal, grouped


# utils

def tv_anisotropic(u):
    """Anisotropic total variation of a 2D array."""
    dx = np.diff(u, axis=0)   # horizontal differences (m-1, n)
    dy = np.diff(u, axis=1)   # vertical differences   (m, n-1)
    return np.sum(np.abs(dx)) + np.sum(np.abs(dy))

def tv_isotropic(u):
    """Isotropic total variation of a 2D array."""
    dx = np.diff(u, axis=0)[:, :-1]   # keep compatible sizes
    dy = np.diff(u, axis=1)[:-1, :]
    grad_mag = np.sqrt(dx**2 + dy**2)
    return np.nansum(grad_mag)

def weight2map(weight, axis_mask, coord=1):
    field_map = reduced2full(weight[None,:], axis_mask)[0, :, coord]
    return field_map.reshape(4,4)

def rmvnan(arr):
    # remove all nan rows and columns
    arr = arr[~np.isnan(arr).all(axis=1)]
    arr = arr[:, ~np.isnan(arr).all(axis=0)]
    return arr