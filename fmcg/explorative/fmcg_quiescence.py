import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def _butter_lowpass(cutoff_hz, fs_hz, order=4):
    nyq = 0.5 * fs_hz
    Wn = cutoff_hz / nyq
    b, a = signal.butter(order, Wn, btype="low", analog=False)
    return b, a


def smooth_qrs_amplitudes(times_s, X_beats, cutoff_hz=0.2, fs_uniform=500, order=4):
    """
    Low-pass filter per-channel QRS amplitudes at ~0.2 Hz (as in the paper), while respecting
    irregular beat times by resampling to a uniform grid, filtering, then sampling back.

    Parameters
    ----------
    times_s : (n_beats,) array
        Beat times in seconds (monotonic increasing).
    X_beats : (n_beats, n_channels) array
        Per-beat QRS amplitudes for each channel.
    cutoff_hz : float
        Low-pass cutoff frequency in Hz. Default 0.2 Hz (per paper).
    fs_uniform : float
        Uniform resampling rate in Hz for filtering stage. Default 4 Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    X_smooth : (n_beats, n_channels) array
        Smoothed per-beat QRS amplitudes.
    """
    times_s = np.asarray(times_s)
    X_beats = np.asarray(X_beats)
    assert (
        times_s.ndim == 1 and X_beats.ndim == 2 and X_beats.shape[0] == times_s.shape[0]
    )

    t0, t1 = times_s[0], times_s[-1]
    if t1 <= t0:
        raise ValueError("times_s must be strictly increasing")
    n_uniform = int(np.ceil((t1 - t0) * fs_uniform)) + 1
    t_uniform = np.linspace(t0, t1, n_uniform)

    # Interpolate to uniform grid, filter, and sample back
    b, a = _butter_lowpass(cutoff_hz, fs_uniform, order)
    X_smooth = np.zeros_like(X_beats, dtype=float)
    for ch in range(X_beats.shape[1]):
        x = X_beats[:, ch]
        xu = np.interp(t_uniform, times_s, x)
        # zero-phase filtering
        xf = signal.filtfilt(b, a, xu, padlen=min(3 * max(len(a), len(b)), len(xu) - 1))
        X_smooth[:, ch] = np.interp(times_s, t_uniform, xf)
    return X_smooth


def compute_signal_space_velocity(times_s, X_smooth):
    """
    Compute ΔX/Δt, its SVD-based rank-reduced & whitened components, and norms.

    Parameters
    ----------
    times_s : (n_beats,) array
    X_smooth : (n_beats, n_channels) array

    Returns
    -------
    rr_s : (n_beats-1,) array
        Beat-to-beat intervals in seconds.
    dX_dt : (n_beats-1, n_channels) array
        Raw ΔX/Δt per channel (no rank-reduction).
    vss_raw : (n_beats-1,) array
        L2 norm of ΔX/Δt across channels (raw).
    Z : (n_beats-1, k) array
        Rank-reduced (90% variance) and whitened components.
    vss_whitened : (n_beats-1,) array
        L2 norm of Z (used for chi-square detection if desired).
    """
    times_s = np.asarray(times_s)
    X_smooth = np.asarray(X_smooth)
    rr_s = np.diff(times_s)
    if np.any(rr_s <= 0):
        raise ValueError("times_s must be strictly increasing")
    dX = np.diff(X_smooth, axis=0)
    dX_dt = dX / rr_s[:, None]
    vss_raw = np.linalg.norm(dX_dt, axis=1)

    # # PCA to get components that explain 90% variance
    # pca = PCA(n_components=None, svd_solver="full")
    # Y = pca.fit_transform(dX_dt)  # shape (n-1, n_channels)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # k = int(np.searchsorted(cumsum, 0.90) + 1)
    # Zk = Y[:, :k]

    # Whiten: zero-mean, unit-variance per component
    Zm = dX_dt - dX_dt.mean(axis=0, keepdims=True)
    std = Zm.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    Z = Zm / std
    vss_whitened = np.linalg.norm(Z, axis=1)
    return rr_s, dX_dt, vss_raw, Z, vss_whitened


def detect_movement(
    times_s, Z, vss_whitened, p_chisq=0.01, ggd_pfp=1e-8, random_state=0
):
    """
    Apply two detectors:
      1) Chi-square on whitened components: v^2 ~ chi2(k), right-tail test at p_chisq.
      2) GMM-based "GGD" on component distribution to approximate low-var noise vs high-var movement.

    Returns
    -------
    det_chi : (n-1,) bool
    det_ggd : (n-1,) bool
    thresholds : dict with 'chi2' and 'ggd' numeric thresholds (on v^2 for chi2, on v for ggd)
    """
    # Chi-square detector on v^2
    v2 = vss_whitened**2
    chi_thresh = stats.chi2.isf(p_chisq, df=3)
    det_chi = v2 > chi_thresh

    # GGD approximation:
    # Fit a 2-component Gaussian mixture to the *components* pooled across dims (as in paper, on ΔX/Δt components).
    # Here we pool Z's components into 1D and fit GMM; pick the lower-variance component as "noise".
    z_flat = Z.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=random_state, n_init=5
    )
    gmm.fit(z_flat)
    vars_ = np.array([np.squeeze(cov) for cov in gmm.covariances_])
    means_ = gmm.means_.squeeze()
    noise_idx = np.argmin(vars_)
    mu0, var0 = means_[noise_idx], vars_[noise_idx]

    # Translate PFP to a threshold on the *norm* v = ||Z|| assuming components ~ N(mu0, var0) i.i.d.
    # Approximate by central case (mu0≈0) => each component ~ N(0, var0), then v^2/var0 ~ chi2_k.
    # Thus set threshold on v by inverse CDF of chi-square at (1-PFP).
    chi2_thresh_noise = stats.chi2.isf(ggd_pfp, df=3)
    ggd_thresh_v = np.sqrt(var0 * chi2_thresh_noise)
    det_ggd = vss_whitened > ggd_thresh_v

    thresholds = {"chi2_v2": chi_thresh, "ggd_v": ggd_thresh_v}
    return det_chi, det_ggd, thresholds


def find_quiescent_segments(
    times_s,
    det_chi,
    det_ggd,
    vss_whitened,
    min_quiet_sec=20.0,
    enter_beats=5,
    exit_beats=3,
):
    """
    Label quiescent beats where neither detector fires and vSS is near-zero, with hysteresis.

    Parameters
    ----------
    min_quiet_sec : float
        Minimum duration to accept a quiescent segment.
    enter_beats : int
        Require this many consecutive non-detections to enter quiescence.
    exit_beats : int
        Require this many consecutive detections to exit quiescence.

    Returns
    -------
    quiet_mask : (n_beats,) bool
        Boolean mask (aligned to beat times) marking quiescent beats.
    segments : list of (t_start, t_end)
        Quiescent segment time spans.
    """
    # Movement detections are defined between beats; extend to beat indices [1..n-1]; pad for beat 0.
    n_beats = len(times_s)
    move = np.zeros(n_beats, dtype=bool)
    move[1:] = det_chi | det_ggd

    quiet = np.zeros(n_beats, dtype=bool)
    in_quiet = False
    consec_no = 0
    consec_yes = 0
    for i in range(n_beats):
        if not move[i]:  # no detection at this beat
            consec_no += 1
            consec_yes = 0
            if not in_quiet and consec_no >= enter_beats:
                in_quiet = True
        else:
            consec_yes += 1
            consec_no = 0
            if in_quiet and consec_yes >= exit_beats:
                in_quiet = False
        quiet[i] = in_quiet

    # Enforce minimum duration
    segments = []
    if n_beats > 0:
        t = times_s
        i = 0
        while i < n_beats:
            if quiet[i]:
                j = i
                while j < n_beats and quiet[j]:
                    j += 1
                if t[j - 1] - t[i] >= min_quiet_sec:
                    segments.append((t[i], t[j - 1]))
                else:
                    quiet[i:j] = False
                i = j
            else:
                i += 1
    return quiet, segments


def compute_baseline_hr(times_s, quiet_mask, hr_bpm, tolerance_bpm=5.0):
    """
    Compute baseline HR from quiescent beats (robust median), and a mask of beats within ±tolerance bpm.

    Returns
    -------
    baseline_bpm : float
    at_baseline_mask : (n_beats,) bool
    """
    hr_bpm = np.asarray(hr_bpm)
    if not np.any(quiet_mask):
        raise ValueError("No quiescent beats found to estimate baseline.")
    baseline_bpm = float(np.median(hr_bpm[quiet_mask]))
    at_baseline_mask = np.abs(hr_bpm - baseline_bpm) <= tolerance_bpm
    return baseline_bpm, at_baseline_mask


def estimate_quiescence_and_baseline(
    times_s,
    X_beats,
    p_chisq=0.01,
    ggd_pfp=1e-8,
    min_quiet_sec=20.0,
    enter_beats=5,
    exit_beats=3,
    cutoff_hz=0.2,
    fs_uniform=4.0,
    plot=False
):
    """
    Full pipeline:
      1) Smooth QRS amplitudes at 0.2 Hz
      2) Build ΔX/Δt, PCA(90%), whiten, vSS
      3) Chi-square + GGD detectors
      4) Quiescent segments with hysteresis + min duration
      5) Baseline HR (median in quiescence) + ±5 bpm mask

    Returns
    -------
    results : dict with keys:
        X_smooth, rr_s, hr_bpm, dX_dt, vss_raw, Z, vss_whitened, k,
        det_chi, det_ggd, thresholds, quiet_mask, segments,
        baseline_bpm, at_baseline_mask
    """
    X_smooth = smooth_qrs_amplitudes(
        times_s, X_beats, cutoff_hz=cutoff_hz, fs_uniform=fs_uniform
    )
    if plot:
        plt.plot(times_s, X_beats, label="Raw")
        plt.plot(times_s, X_smooth, label="Smoothed")
        plt.legend()
        plt.show()

    rr_s = np.diff(times_s)
    hr_bpm = 60.0 / rr_s
    rr_s, dX_dt, vss_raw, Z, vss_whitened = compute_signal_space_velocity(
        times_s, X_smooth
    )

    if plot:
        plt.plot(vss_whitened / vss_whitened.max(), label="vSS (whitened)")
        plt.plot(vss_raw / vss_raw.max(), label="vSS (raw)")
        plt.legend()
        plt.show()

    det_chi, det_ggd, thresholds = detect_movement(
        times_s, Z, vss_whitened, p_chisq=p_chisq, ggd_pfp=ggd_pfp
    )

    quiet_mask, segments = find_quiescent_segments(
        times_s,
        det_chi,
        det_ggd,
        vss_whitened,
        min_quiet_sec=min_quiet_sec,
        enter_beats=enter_beats,
        exit_beats=exit_beats,
    )
    # For HR per beat, pad last value to align (beats vs intervals)
    hr_bpm_beats = np.r_[hr_bpm, hr_bpm[-1]]
    baseline_bpm, at_baseline_mask = compute_baseline_hr(
        times_s, quiet_mask, hr_bpm_beats, tolerance_bpm=5.0
    )

    return {
        "X_smooth": X_smooth,
        "rr_s": rr_s,
        "hr_bpm": hr_bpm_beats,
        "dX_dt": dX_dt,
        "vss_raw": vss_raw,
        "Z": Z,
        "vss_whitened": vss_whitened,
        "det_chi": det_chi,
        "det_ggd": det_ggd,
        "thresholds": thresholds,
        "quiet_mask": quiet_mask,
        "segments": segments,
        "baseline_bpm": baseline_bpm,
        "at_baseline_mask": at_baseline_mask,
    }


def plot_quiescence_results(times_s, results, figsize=(12, 6)):
    """
    Plot FHR trace with baseline and quiescence, plus vSS with detections.
    """
    hr = results["hr_bpm"]
    baseline = results["baseline_bpm"]
    quiet_mask = results["quiet_mask"]
    at_baseline = results["at_baseline_mask"]
    vss = results["vss_whitened"]
    det_chi = results["det_chi"]
    det_ggd = results["det_ggd"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- HR trace ---
    ax1.plot(times_s, hr, color="black", lw=1, label="HR (bpm)")
    ax1.axhline(baseline, color="blue", ls="--", label=f"Baseline={baseline:.1f} bpm")
    ax1.fill_between(
        times_s,
        hr.min() - 10,
        hr.max() + 10,
        where=quiet_mask,
        color="green",
        alpha=0.2,
        step="mid",
        label="Quiescence",
    )
    ax1.fill_between(
        times_s,
        hr.min() - 10,
        hr.max() + 10,
        where=at_baseline,
        color="blue",
        alpha=0.1,
        step="mid",
        label="At baseline ±5 bpm",
    )
    ax1.set_ylabel("HR (bpm)")
    ax1.legend(loc="upper right")

    # --- vSS trace ---
    t_mid = (times_s[:-1] + times_s[1:]) / 2
    ax2.plot(t_mid, vss, color="gray", lw=1, label="vSS (whitened)")
    ax2.scatter(
        t_mid[det_chi],
        vss[det_chi],
        color="red",
        s=20,
        marker="x",
        label="Chi2 detects",
    )
    ax2.scatter(
        t_mid[det_ggd],
        vss[det_ggd],
        color="purple",
        s=20,
        marker="o",
        facecolors="none",
        label="GGD detects",
    )
    ax2.set_ylabel("vSS (a.u.)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig
