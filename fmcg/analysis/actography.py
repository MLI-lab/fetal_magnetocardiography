from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_closing


def compute_actogram(
    signal,
    peaks,
    fs,
    variance_threshold=0.90,
    pfp=1e-8,
    min_gap_beats=4,
    reference_mask=None,
    plot=True,
):
    """
    vSS actography following Lutter & Wakai (2011).

    Parameters
    ----------
    signal : (n_samples, n_channels)
        Multichannel MCG signal
    peaks : (n_beats,)
        Sample indices of detected fetal R-peaks
    fs : float
        Sampling frequency in Hz
    variance_threshold : float
        Cumulative power threshold for SVD rank reduction (default 0.90)
    pfp : float
        Probability of false positive for detection threshold (default 1e-8)
    min_gap_beats : int
        Minimum gap between detections in beats. Merges detections separated
        by fewer than this many beats. Set to 0 to disable merging (default 4)
    reference_mask : (n_beats-1,) boolean array, optional
        Mask indicating quiescent reference periods for noise estimation
    plot : bool
        Whether to plot the resulting actogram (default True)

    Returns
    -------
    vss : (n_beats-1,)
        Scalar velocity signal-space actogram
    time_axis : (n_beats-1,)
        Time in seconds corresponding to each vSS sample
    detections : (n_beats-1,) boolean array
        True where movement detected based on GMM detector

    References
    ----------
    Lutter, William J., and Ronald T. Wakai. ‘Indices and Detectors for Fetal MCG Actography’. IEEE Transactions on Bio-Medical Engineering 58, no. 6 (2011): 1874–80. https://doi.org/10.1109/TBME.2011.2131141.

    """
    # Compute velocity signal-space (vSS)
    vss, V, time_axis = _vss(signal, peaks, fs)

    # Rank reduction and whitening
    Z, k = _rank_reduce_and_whiten(V, variance_threshold, reference_mask)

    # Apply detector
    detections, gmm, T, threshold = _fit_ggd_multivariate(Z, pfp)

    if min_gap_beats > 0:
        detections = binary_closing(
            detections, structure=np.ones(2 * min_gap_beats + 1, dtype=bool)
        )

    if plot:
        _plot_actography(time_axis, vss, detections, title=f"Actogram", log=False)

    return vss, time_axis, detections


def _vss(signal, peaks, fs):
    """
    Signal-space velocity (vSS) calculation for fetal movement detection.

    Lutter & Wakai (2011): "The velocity signal-space index is defined as
    vSS[n] = |ΔX[n]| / Δt[n], where ΔX[n] = X[n] - X[n-1] is the beat-to-beat
    change in the signal-space vector and Δt[n] is the RR interval."

    Parameters
    ----------
    signal : (n_samples, n_channels)
        Multichannel signal
    peaks : (n_beats,)
        Sample indices of R-peaks
    fs : float
        Sampling frequency

    Returns
    -------
    vss : (n_beats-1,)
        Scalar actogram (velocity magnitude)
    V : (n_beats-1, n_channels)
        Velocity vectors
    time_axis : (n_beats-1,)
        Time in seconds for each velocity sample
    """
    # Signal-space vectors at each beat: X[n]
    X = signal[peaks]

    # Beat-to-beat difference: ΔX[n] = X[n] - X[n-1]
    dX = np.diff(X, axis=0)

    # RR intervals in seconds: Δt[n]
    rr = np.diff(peaks) / fs

    # Velocity vectors: V[n] = ΔX[n] / Δt[n]
    V = dX / (rr[:, None] + 1e-12)

    # Scalar actogram: vSS[n] = |V[n]|
    vss = np.linalg.norm(V, axis=1)

    # Time axis (use middle of RR interval)
    time_axis = peaks[1:] / fs

    return vss, V, time_axis


def _rank_reduce_and_whiten(V, variance_threshold, reference_mask):
    """
    SVD rank reduction and whitening of velocity matrix.

    Lutter & Wakai (2011): "Whitening was accomplished by normalizing each
    component of vSS by the variance of that component."

    Parameters
    ----------
    V : (n_beats-1, n_channels)
        Velocity matrix
    variance_threshold : float
        Retain components explaining this fraction of total power
    reference_mask : (n_beats-1,) boolean array, optional
        Mask for quiescent reference window

    Returns
    -------
    Z : (n_beats-1, k)
        Whitened, rank-reduced velocity projections
    k : int
        Effective rank (number of retained components)
    """
    # SVD: V = U @ diag(s) @ Wt
    U, s, Wt = np.linalg.svd(V, full_matrices=False)

    # Determine rank k from power threshold
    power = s**2
    cumulative_power = np.cumsum(power) / power.sum()
    k = int(np.searchsorted(cumulative_power, variance_threshold)) + 1

    # Project onto top-k principal components
    Z = U[:, :k]  # (n_beats-1, k)

    # Estimate noise variance from quiescent reference window
    # "The variance of each component was estimated from periods of quiescence"
    if reference_mask is not None and reference_mask.sum() > 0:
        ref_data = Z[reference_mask]
    else:
        ref_data = Z

    # Compute standard deviation per component
    sigma = ref_data.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)  # Avoid division by zero

    # Whiten to unit variance: each component should have variance = 1 under H0
    Z_whitened = Z / sigma

    return Z_whitened, k


def _fit_ggd_multivariate(
    Z: np.ndarray,
    pfp: float = 1e-8,
) -> tuple[np.ndarray, GaussianMixture, np.ndarray, np.ndarray]:
    """
    GGD following Kay (1998) — GMM fitted on the k-dimensional whitened
    velocity vectors Z directly, using isotropic (spherical) covariance.

    Under H0: z ~ N(0, sigma0^2 * I_k)   [quiescence]
    Under H1: z ~ N(0, sigma1^2 * I_k)   [movement]

    The log-likelihood ratio is monotone in T = ||z||^2, whose distribution
    under H0 is sigma0^2 * chi^2(k).  The threshold on T is set analytically
    from the noise component variance sigma0^2 and the target PFP.

    Parameters
    ----------
    Z   : (n_beats, k)  whitened, rank-reduced velocity projections
    pfp : target probability of false positive under H0

    Returns
    -------
    detections : (n_beats,) bool — True where movement detected
    gmm        : fitted GaussianMixture
    T          : (n_beats,) scalar test statistic ||z||^2
    log_lr     : (n_beats,) log-likelihood ratio per beat
    """
    n_beats, k = Z.shape

    # --- 1. Fit isotropic 2-component GMM on the full k-dimensional vectors ---
    # covariance_type='spherical' enforces sigma^2 * I structure per component,
    # consistent with the physical model (no preferred movement direction).
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="spherical",
        max_iter=300,
        n_init=10,
        random_state=0,
    )
    gmm.fit(Z)

    # --- 2. Identify noise vs. signal component by variance magnitude ---
    # gmm.covariances_ has shape (2,) for spherical — one scalar sigma^2 per component
    sigma2 = gmm.covariances_  # (2,)  each is the scalar variance
    noise_idx = int(np.argmin(sigma2))
    signal_idx = 1 - noise_idx

    sigma0_sq = sigma2[noise_idx]
    sigma1_sq = sigma2[signal_idx]
    alpha_noise = gmm.weights_[noise_idx]
    alpha_signal = gmm.weights_[signal_idx]

    # --- 3. Compute log-likelihood ratio analytically ---
    # log Λ(z) = log(alpha1/alpha0) - k/2 * log(sigma1^2/sigma0^2)
    #            + ||z||^2 / 2 * (1/sigma0^2 - 1/sigma1^2)
    T = np.sum(Z**2, axis=1)  # (n_beats,)  scalar test statistic

    log_prior_odds = np.log(alpha_signal / (alpha_noise + 1e-300))
    log_var_ratio = (k / 2.0) * np.log(sigma1_sq / (sigma0_sq + 1e-300))
    quadratic_coeff = 0.5 * (1.0 / (sigma0_sq + 1e-300) - 1.0 / (sigma1_sq + 1e-300))

    log_lr = log_prior_odds - log_var_ratio + quadratic_coeff * T

    # --- 4. Set detection threshold from H0 distribution of T ---
    # Under H0:  T / sigma0^2  ~  chi^2(k)
    # So T ~ sigma0^2 * chi^2(k)
    # P(T > eta | H0) = PFP  =>  eta = sigma0^2 * chi2.ppf(1 - pfp, df=k)
    # We translate this threshold into log-LR space for consistency.
    T_threshold = sigma0_sq * stats.chi2.ppf(1.0 - pfp, df=k)
    log_lr_threshold = log_prior_odds - log_var_ratio + quadratic_coeff * T_threshold

    detections = log_lr > log_lr_threshold

    return detections, gmm, T, log_lr


def _plot_actography(
    x_positions, values, detection=None, title="", ylabel="Velocity", log=False, ax=None
):
    """Internal helper for visualization."""
    if ax is None:
        fig = plt.figure(figsize=(7.11, 2), dpi=300)
        ax = fig.gca()
        show_plot = True
    else:
        show_plot = False

    # Main Line
    ax.plot(
        x_positions,
        values,
        ".-",
        label=ylabel,
        color="k",
        fillstyle="none",
        linewidth=0.75,
    )

    # Highlight points
    if detection is not None:
        ax.plot(
            x_positions[detection],
            values[detection],
            ".",
            color="tab:red",
            markersize=3,
            label="Detected Movement",
        )

    # Styling
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.9)
    if log:
        ax.set_yscale("log")

    # x ticks every 10 seconds
    ax.set_xticks(np.arange(0, x_positions[-1], 15))

    ax.legend(ncol=2)

    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax
