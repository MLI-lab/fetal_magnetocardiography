import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import neurokit2 as nk
from scipy.stats import kurtosis
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.discriminant_analysis import StandardScaler
from scipy.signal import correlate
from scipy.stats import zscore
from scipy.fft import rfft, irfft

from fmcg.analysis.heartbeats import detect_hr_outlier


def compute_ICA_statistics(sources, fs, offset=10, span=5):
    """
    Computes statistics for all ICA components.
    """
    n_components = sources.shape[1]
    stats_list = []
    
    # 1. Calculate statistics for ALL components
    for j in range(n_components):
        signal = sources[int(offset*fs) : int(offset*fs + span*fs), j]
        
        # Basic Kurtosis
        kurt = kurtosis(signal)
        
        # Signal processing for peak detection
        try:
            # Invert if necessary and clean
            signal_inv, _ = nk.ecg_invert(signal, fs)
            signal_clean = nk.ecg_clean(signal_inv, sampling_rate=fs, method="vg")
            peaks_dict = nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")
            peaks = peaks_dict["ECG_R_Peaks"]
            
            if len(peaks) > 1:
                # Stats computations               
                snr = 10 * np.log10(np.mean(signal[peaks]**2) / np.mean(signal**2))
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

        stats_list.append({
            'index': j,
            'signal': signal,
            'peaks': peaks,
            'num_peaks': len(peaks),
            'kurtosis': kurt,
            'mean_hr': mean_hr,
            'snr': snr,
            'sdnn': sdnn,
            'hr_outlier_rate': hr_outlier_rate
        })

    # 2. Sort components by SNR (highest SNR first)
    # If you want fetal (usually higher SNR) at the top, keep reverse=True
    stats_list.sort(key=lambda x: x['snr'], reverse=True)

    return stats_list

def _lagged_correlation_matrix(signals):
    """
    Computes a symmetric distance matrix based on max correlation (lag >= 0)
    using vectorized FFT broadcasting.
    """
    n_comps, n_samples = signals.shape
    # 1. Standardize signals along the time axis
    # This ensures the cross-correlation equals the correlation coefficient
    norm_signals = zscore(signals, axis=1)
    
    # 2. Pad for FFT to avoid circular convolution (length 2n - 1)
    pad_len = 2 * n_samples - 1
    # Use rfft for real-valued signals (faster and uses less memory)
    signals_fft = np.fft.rfft(norm_signals, n=pad_len, axis=1)
    
    max_corrs = np.zeros((n_comps, n_comps))

    for i in range(n_comps):
        # 3. Multiply spectrum of signal 'i' with ALL other signals
        # np.conj handles the 'correlation' vs 'convolution' logic
        combined_fft = signals_fft[i] * np.conj(signals_fft)
        
        # 4. Batch Inverse FFT to get all correlations for this row
        # This returns an (n_comps, pad_len) matrix
        corrs = np.fft.irfft(combined_fft, n=pad_len, axis=1)
        
        # 5. Extract lags >= 0 
        # In the 'full' correlation result, lag 0 is at index 0 
        # due to how np.fft.irfft aligns the output.
        # However, to match 'scipy.signal.correlate' logic:
        # Lags [0...n_samples-1] are at the start of the array.
        lags_ge_zero = corrs[:, :n_samples]
        
        # 6. Find max absolute correlation for each pair
        # We divide by n_samples to normalize the correlation to [-1, 1]
        max_corrs[i, :] = np.max(np.abs(lags_ge_zero), axis=1) / n_samples

    # 7. Make the matrix symmetric
    # Since we want distance(A,B) == distance(B,A), we take the 
    # maximum correlation found regardless of which signal was 'leading'.
    max_corrs = np.maximum(max_corrs, max_corrs.T)
    
    return 1 - max_corrs

def _cluster_components(ica_stats, n_components=5, n_clusters=3):
    """
    Subroutine for grouping top n_components into clusters and heuristics-based assignment.
    """
    df = pd.DataFrame(ica_stats).iloc[:n_components]
    features = ['kurtosis', 'mean_hr', 'snr', 'sdnn', 'hr_outlier_rate', 'num_peaks']
    
    # # Clustering 
    # X = df[['mean_hr']]
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # df['cluster'] = kmeans.fit_predict(X_scaled)

    ica_signals = np.array([e['signal'] for e in ica_stats[:n_components]])
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='precomputed', 
        linkage='complete'
    )
    df['cluster'] = cluster_model.fit_predict(_lagged_correlation_matrix(ica_signals))

    # Assign heuristic
    df_cluster = df.groupby('cluster')[features].agg({'mean_hr': 'mean', 'sdnn': 'min', 'snr': 'max'}).sort_values(['mean_hr', "sdnn", "snr"], ascending=[False, True, False]).reset_index()
    fetal = df[df.cluster == df_cluster.loc[0].cluster]['index'].iloc[0]
    maternal = df[df.cluster == df_cluster.loc[1].cluster]['index'].iloc[0]


    df['assigned_label'] = df['cluster'].map({
        df_cluster.loc[0].cluster: 'Fetal',
        df_cluster.loc[1].cluster: 'Maternal'
    }).fillna('Other')

    # sanity check
    # maternal hr should be lower
    if df[df['index'] == maternal].mean_hr.mean() >= df[df['index'] == fetal].mean_hr.mean():
        print("Warning: Heuristic assignment may be incorrect based on mean HR. Please review the clustering results.")
        print(f"Maternal mean HR: {df[df['index'] == maternal].mean_hr.mean()}")
        print(f"Fetal mean HR: {df[df['index'] == fetal].mean_hr.mean()}")
    print(df_cluster)


    return maternal, fetal, df


def select_ica_components(sources, fs, offset=10, plot_span=5, span=5, method="clustering_heuristic", selection=None, clustering_args=None, plot_components=5, plot=False):
    """"
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
    Returns:
        maternal (int): Index of the selected maternal component.
        fetal (int): Index of the selected fetal component.
    """
    if clustering_args is None:
        clustering_args = {"n_components": 5, "n_clusters": 3}
        
    # 1. Compute stats
    ica_stats = compute_ICA_statistics(sources, fs, offset=offset, span=span)

    # 2. Assignment
    if method == "clustering_heuristic":
        maternal, fetal, df_clustered = _cluster_components(ica_stats, **clustering_args)
        cluster_assignment = dict(zip(df_clustered['index'], df_clustered['assigned_label']))
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
            plot_components, 1, 
            sharex=True, 
            figsize=(12, 1.5 * plot_components), 
            dpi=100
        )
        if plot_components == 1: axes = [axes]
        
        for i, comp in enumerate(stats_list):
            ax = axes[i]
            sig = comp['signal'][:int(plot_span*fs)]
            pks = comp['peaks']
            pks = pks[pks < int(plot_span*fs)]
            idx = comp['index']
            
            # Determine coloring
            edge_color = 'lightgrey'
            title_suffix = ""
            is_selected = False
            
            if idx == maternal:
                title_suffix = " (Maternal)"
                is_selected = True
                edge_color = 'tab:blue'
                line_color = 'tab:blue'
            elif idx == fetal:
                title_suffix = " (Fetal)"
                is_selected = True
                edge_color = 'tab:red'
                line_color = 'tab:red'
            else:
                line_color = 'tab:grey'
            
            # Plot Signal
            time_axis = np.arange(plot_span*fs) / fs
            ax.plot(time_axis, sig, label=f"Comp {idx}", color=line_color, lw=1)
            
            # Plot Peaks
            if len(pks) > 0:
                ax.plot(pks / fs, sig[pks], "rx", markersize=7, label="Peaks")

            # Annotation Box
            stats_text = (f"SDNN: {comp['sdnn']:.2f} ms\n"
                          f"Kurt: {comp['kurtosis']:.2f}\n"
                          f"SNR: {comp['snr']:.2f} dB\n"
                          f"Mean HR: {comp['mean_hr']:.2f} BPM\n"
                          f"Num Peaks: {len(pks)}\n"
                          f"HR Outlier Rate: {comp['hr_outlier_rate']:.2f}")
            
            ax.text(1.01, 0.5, stats_text, transform=ax.transAxes, 
                    va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
            
            # Title
            title_str = f"ICA Component {idx}"
            if method == "clustering_heuristic" and idx in cluster_assignment:
                title_str += f" - Cluster {cluster_assignment[idx]}"
            title_str += title_suffix
            
            ax.set_title(title_str, fontsize=12, fontweight='bold' if is_selected else 'normal')
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

    return maternal, fetal, cluster_assignment


def plot_similarity_with_energy_context(Um, Uf, Lambda_m, Lambda_f,
                                              k_m=None, k_f=None,
                                              metric='angle',
                                              names=['Maternal', 'Fetal']):
    # 1. Setup & Computations
    if k_m is None: k_m = Um.shape[1]
    if k_f is None: k_f = Uf.shape[1]
    
    Um_k, Uf_k = Um[:, :k_m], Uf[:, :k_f]
    Lambda_m_k, Lambda_f_k = Lambda_m[:k_m], Lambda_f[:k_f]
    
    # Calculate Similarity (Alignment)
    S = np.abs(Um_k.T @ Uf_k)
    # Calculate Angles for the heatmap and the min-angle lines
    angles = np.rad2deg(np.arccos(np.clip(S, 0, 1)))
    
    if metric == 'angle':
        data = angles
        vmin, vmax = 0, 90
        cbar_label = 'Angle θ (degrees)'
        fmt = '{:.0f}°'
    else: 
        data = S
        vmin, vmax = 0, 1
        cbar_label = 'Alignment (cos θ)'
        fmt = '{:.2f}'
        
    # Min angles per component against the other subspace
    min_angles_m = np.min(angles[:k_m, :k_f], axis=1) # Min angle for each Maternal comp
    min_angles_f = np.min(angles[:k_m, :k_f], axis=0) # Min angle for each Fetal comp

    energy_m = (Lambda_m_k / np.sum(Lambda_m_k)) * 100
    energy_f = (Lambda_f_k / np.sum(Lambda_f_k)) * 100

    # 2. Figure Layout
    fig = plt.figure(figsize=(4.5, 4.5))
    gs = GridSpec(2, 3, width_ratios=[5, 1.5, 0.2], height_ratios=[1.5, 5], 
                  wspace=0.1, hspace=0.1)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])
    
    # 3. Heatmap
    im = ax_main.imshow(data, cmap='plasma', aspect='equal', vmin=vmin, vmax=vmax)
    
    if k_m <= 15 and k_f <= 15:
        thresh = vmin + (vmax - vmin) / 2.5 
        for i in range(k_m):
            for j in range(k_f):
                val = data[i, j]
                dark_bg = val > thresh if metric != 'angle' else val < thresh
                ax_main.text(j, i, fmt.format(val), ha='center', va='center', 
                             color='white' if dark_bg else 'black', fontsize=7)

    ax_main.set_xticks(np.arange(k_f))
    ax_main.set_yticks(np.arange(k_m))
    ax_main.set_xticklabels([str(j+1) for j in range(k_f)])
    ax_main.set_yticklabels([str(i+1) for i in range(k_m)])
    ax_main.set_xlabel(f'{names[1]} Component', fontsize=10)
    ax_main.set_ylabel(f'{names[0]} Component', fontsize=10)
    ax_main.grid(which='minor', color='white', linestyle='-', linewidth=1)
    for spine in ax_main.spines.values(): spine.set_visible(False)

    # 4. Top Marginal (Fetal): Bars = Energy, Line = Min Angle
    bars_top = ax_top.bar(np.arange(k_f), energy_f, color='lightgrey', width=0.7)
    ax_top.bar_label(bars_top, fmt='%.0f%%', padding=1, fontsize=6)
    ax_top_angle = ax_top.twinx()
    ax_top_angle.plot(np.arange(k_f), min_angles_f, color='crimson', marker='o', markersize=3, linewidth=1)
    ax_top_angle.set_ylim(0, 90)
    ax_top_angle.set_yticks([0, 30, 60, 90])
    ax_top_angle.set_ylabel('Min θ (°)', color='crimson', fontsize=8)
    ax_top_angle.tick_params(axis='y', labelcolor='crimson', labelsize=7)
    ax_top.set_ylabel('Energy %', fontsize=8)
    ax_top.tick_params(labelbottom=False, bottom=False, labelsize=7)

    # 5. Right Marginal (Maternal): Bars = Energy, Line = Min Angle
    bars_right = ax_right.barh(np.arange(k_m), energy_m, color='lightgrey', height=0.7)
    ax_right.bar_label(bars_right, fmt='%.0f%%', padding=1, fontsize=6, rotation=-90)
    ax_right_angle = ax_right.twiny()
    ax_right_angle.plot(min_angles_m, np.arange(k_m), color='crimson', marker='o', markersize=3, linewidth=1)
    ax_right_angle.set_xlim(0, 90)
    ax_right_angle.set_xticks([0, 30, 60, 90])
    ax_right_angle.set_xlabel('Min θ (°)', color='crimson', fontsize=8)
    ax_right_angle.tick_params(axis='x', labelcolor='crimson', labelsize=7)
    ax_right.set_xlabel('Energy %', fontsize=8)
    ax_right.tick_params(labelleft=False, left=False, labelsize=7)

    # 6. Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=9, labelpad=10, rotation=270, va='bottom')
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    return fig, S