import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fmcg.ics.functions import plotHR, avg_channels_hr_range
from fmcg.utils.plotting.plot_utils import plot_sensor_signals
from fmcg.utils.utils import reduced2full

def construct_template(data, sources, fs, comps, window_size=0.25, bpm_tol=5, plot=False, axis_mask=None):
    assert data.ndim == 2

    # Compute template from average of heartbeats
    heartRate, peaks = plotHR(sources, comps, fs=fs, plot=plot)

    # average the segments from the remaining field measurements after maternal cancelation
    avg_waveform, std_waveform = avg_channels_hr_range(
        data,
        peaks,
        heartRate,
        window_size=window_size,
        denoise=False,
        plot=plot,
        min_hr=np.median(heartRate) - bpm_tol,
        max_hr=np.median(heartRate) + bpm_tol,
        plt_save=False,
    )
    topography =  np.array(avg_waveform).T

    if axis_mask is not None and plot:
        full_topography = reduced2full(topography, axis_mask)

        plot_sensor_signals(
            full_topography,
            np.arange(full_topography.shape[0]) / fs,
            ylabel="Reconstructed Field [pT]",
            xlim=[0, full_topography.shape[0] / fs],
            sharey=True,
        )
    if axis_mask is None and plot:
        raise ValueError("To plot the topography, please provide an axis_mask indicating the sensor locations.")

    return topography

def lmmse(data, topography):
    # LMMSE Filter C_T / C_S
    signal_cov = np.cov(data.T)
    U, S, _ = np.linalg.svd(signal_cov, hermitian=True)
    W = np.cov(topography.T) @ U @ np.diag(1/S) @ U.T

    return data @ W.T

def ssp(data, topography, k=None, explained_variance=None):
    if k is not None and explained_variance is not None:
        raise ValueError("Please provide either `k` or `explained_variance`, not both.")
    if k is None and explained_variance is None:
        raise ValueError("Please provide either `k` or `explained_variance`.")

    # SSP Filter
    template = np.cov(topography.T)
    U, S, _ = np.linalg.svd(template, hermitian=True)
    explained_variance_by_component = S / np.sum(S)
    
    with np.printoptions(precision=2, floatmode="fixed", suppress=True):
        print(f"Explained Variance by each component: {explained_variance_by_component*100}%")
        print(f"Cumulative Explained Variance: {np.cumsum(explained_variance_by_component)*100}%")
        print(f"Rank of template covariance: {np.linalg.matrix_rank(template)}")
    if k is None:
        k = np.searchsorted(np.cumsum(explained_variance_by_component), explained_variance) + 1
        print(f"Selected number of components: {k}, explaining {np.cumsum(explained_variance_by_component)[k-1]*100: .2f}% of variance")

    W = np.eye(U.shape[0]) - (U[:, :k] @ U[:, :k].T)
    
    return data @ W.T

def plot_similarity_with_energy_context(topography_m, topography_f,
                                              k_m=None, k_f=None,
                                              metric='angle',
                                              names=['Maternal', 'Fetal']):
    # 1. Setup & Computations
    if k_m is None: k_m = Um.shape[1]
    if k_f is None: k_f = Uf.shape[1]

    # Get eigenvectors from covariance matrices
    # Assuming avg_waveform_m shape: (n_sensors, n_timepoints)
    Cm = np.cov(topography_m.T)
    Cf = np.cov(topography_f.T)

    # Eigendecomposition (sorted by eigenvalue)
    eigenvalues_m, eigenvectors_m = np.linalg.eigh(Cm)
    eigenvalues_f, eigenvectors_f = np.linalg.eigh(Cf)

    # Sort descending
    idx_m = np.argsort(eigenvalues_m)[::-1]
    idx_f = np.argsort(eigenvalues_f)[::-1]

    Um = eigenvectors_m[:, idx_m]  # shape: (n_sensors, n_sensors)
    Uf = eigenvectors_f[:, idx_f]
    Lambda_m = eigenvalues_m[idx_m]
    Lambda_f = eigenvalues_f[idx_f]
    
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