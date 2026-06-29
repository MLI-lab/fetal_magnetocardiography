"""Vectorcardiogram (VCG) loop analysis.

General-purpose tools for analyzing a 3D cardiac moment trajectory ``moment`` (shape ``(T, 3)``,
e.g. the fetal dipole moment from a forward-model fit): per-beat peak cardiac vector, orientation
and magnitude drift over time, minimal-rotation alignment, and the low-rank (SVD) drift subspace.

Loop extraction itself is just heartbeat averaging — use
:func:`fmcg.analysis.heartbeat_averaging.average_heartbeats` / ``extract_epochs`` to build loops,
then pass them here for plotting / SVD.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- geometry helpers
def axis_rotation(a, b):
    """Minimal rotation matrix taking unit vector ``a`` to unit vector ``b`` (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:                       # parallel or anti-parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _moving_average(x, w):
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(int(w)) / int(w)
    return np.convolve(np.pad(x, int(w) // 2, mode="edge"), k, "valid")[: len(x)]


# --------------------------------------------------------------------------- peak vector & drift
def peak_vectors(moment, peaks, fs, half_ms=50.0):
    """Per-beat peak cardiac vector: the sample of largest ``|moment|`` within ±``half_ms`` of each peak.

    Parameters
    ----------
    moment : (T, 3) array — 3D cardiac moment trajectory.
    peaks  : (n_beats,) int array — R-peak sample indices.
    fs     : float — sampling rate [Hz].
    half_ms: float — half search window around each peak [ms].

    Returns
    -------
    (n_beats, 3) array of peak cardiac vectors.
    """
    moment = np.asarray(moment)
    half = max(1, int(half_ms * 1e-3 * fs))
    T = moment.shape[0]
    out = np.empty((len(peaks), 3))
    for i, p in enumerate(peaks):
        seg = moment[max(0, p - half): min(T, p + half)]
        out[i] = seg[np.argmax(np.linalg.norm(seg, axis=1))]
    return out


def vcg_drift(moment, peaks, fs, half_ms=50.0, smooth_beats=21):
    """Per-beat VCG orientation and magnitude drift relative to a robust reference.

    The peak cardiac vector is taken per beat; vectors are sign-aligned (resolving the R/S
    ~180° phase ambiguity) to a robust mean direction. Orientation drift is the angle to that
    reference direction; magnitude drift is the peak magnitude relative to its median.

    Returns a dict with: ``beat_t``, ``peak_vecs``, ``units`` (sign-aligned), ``mags``,
    ``mean_dir``, ``median_mag``, ``angle_deg``, ``scale`` (mags / median), and the
    smoothed ``angle_deg_smooth`` / ``scale_smooth``.
    """
    peaks = np.asarray(peaks)
    pv = peak_vectors(moment, peaks, fs, half_ms)
    mags = np.linalg.norm(pv, axis=1)
    units = pv / mags[:, None]

    mean_dir = np.median(units, axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    for _ in range(3):                              # resolve sign flips, refine reference
        units = units * np.sign(units @ mean_dir)[:, None]
        mean_dir = units.mean(0)
        mean_dir /= np.linalg.norm(mean_dir)

    angle = np.degrees(np.arccos(np.clip(units @ mean_dir, -1.0, 1.0)))
    median_mag = float(np.median(mags))
    scale = mags / median_mag
    # Smoothed 3-D orientation trajectory: the per-beat ``units`` carry large measurement noise
    # (~8 deg/beat) that swamps the slow real drift. Smooth each component over ``smooth_beats`` and
    # renormalize (small-angle geodesic mean) so a replay reproduces the slow drift, not the noise;
    # regime shifts spanning many beats survive the window.
    units_smooth = np.column_stack([_moving_average(units[:, c], smooth_beats) for c in range(3)])
    units_smooth /= np.linalg.norm(units_smooth, axis=1, keepdims=True) + 1e-12
    return dict(
        beat_t=peaks / fs, peak_vecs=pv, units=units, mags=mags,
        mean_dir=mean_dir, median_mag=median_mag, angle_deg=angle, scale=scale,
        units_smooth=units_smooth,
        angle_deg_smooth=_moving_average(angle, smooth_beats),
        scale_smooth=_moving_average(scale, smooth_beats),
    )


def svd_drift_modes(epochs, global_mean=None, n_modes=3):
    """Low-rank drift subspace: SVD of mean-subtracted epochs.

    The leading temporal scores are the beat-to-beat trajectories of the dominant nonstationary
    modes — expected to track :func:`vcg_drift` orientation/magnitude.

    Parameters
    ----------
    epochs : (n_beats, L, C) or (n_beats, L*C) array of per-beat loops/epochs.
    global_mean : optional mean to subtract (default: ``epochs.mean(0)``).
    n_modes : number of leading modes to return.

    Returns
    -------
    dict with ``scores`` (n_beats, n_modes), ``components`` (n_modes, L*C),
    ``singular_values`` (n_modes,), ``explained_var`` (n_modes,).
    """
    X = np.asarray(epochs)
    X = X.reshape(X.shape[0], -1)
    mu = X.mean(0) if global_mean is None else np.asarray(global_mean).ravel()
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    return dict(
        scores=U[:, :n_modes] * S[:n_modes], components=Vt[:n_modes],
        singular_values=S[:n_modes], explained_var=(var[:n_modes] / var.sum()),
    )


# --------------------------------------------------------------------------- plotting
def plot_vcg_drift(drift, axes=None, figsize=(7.11, 4.0)):
    """Plot orientation and magnitude drift over recording time (per beat + smoothed)."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    t = drift["beat_t"]
    axes[0].plot(t, drift["angle_deg"], ".", ms=2, alpha=0.3, color="C0", label="per beat")
    axes[0].plot(t, drift["angle_deg_smooth"], "C0", lw=2, label="smoothed")
    axes[0].set_ylabel("orientation drift [deg]"); axes[0].legend()
    axes[1].plot(t, drift["scale"], ".", ms=2, alpha=0.3, color="C1")
    axes[1].plot(t, drift["scale_smooth"], "C1", lw=2)
    axes[1].set_ylabel("magnitude (rel.)"); axes[1].set_xlabel("recording time [s]")
    for a in axes:
        a.grid(alpha=0.3)
    return axes


def plot_vcg_loops(loops, labels=None, colors=None, scale=1.0, axes=None, figsize=(7.11, 2.5)):
    """Plot a list of 3D loops ``(L, 3)`` as a 3D trajectory + xy/xz/yz projections.

    ``loops`` are already-averaged loops (build them with ``average_heartbeats``); this only plots.
    Paper layout (7.11in, equal-aspect projections); the dense 4-panel uses small tick/legend fonts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    created = axes is None
    if created:
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1.5, 1, 1, 1], wspace=0.35)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = [fig.add_subplot(gs[0, k + 1]) for k in range(3)]
    else:
        ax3d, ax2d = axes[0], axes[1:]
    colors = colors or [f"C{i}" for i in range(len(loops))]
    planes = [((0, 1), "$x$", "$y$"), ((0, 2), "$x$", "$z$"), ((1, 2), "$y$", "$z$")]
    for i, lp in enumerate(loops):
        b = np.asarray(lp) * scale
        lbl = labels[i] if labels is not None else None
        ax3d.plot(b[:, 0], b[:, 1], b[:, 2], color=colors[i], lw=1, alpha=0.7, label=lbl)
    ax3d.set_xlabel("$x$", labelpad=-5); ax3d.set_ylabel("$y$", labelpad=-5)
    ax3d.set_zlabel("$z$", labelpad=-5)
    ax3d.set_title("3D Trajectory", y=0.95, fontsize=8)
    ax3d.view_init(elev=30, azim=240); ax3d.tick_params(labelsize=7)
    for a in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        a.set_major_locator(MaxNLocator(nbins=3))
    if labels is not None:
        ax3d.legend(loc="upper center", bbox_to_anchor=(2.1, 0), ncol=3, fontsize=7)
    for ax, ((xi, yi), xl, yl) in zip(ax2d, planes):
        for i, lp in enumerate(loops):
            b = np.asarray(lp) * scale
            ax.plot(b[:, xi], b[:, yi], color=colors[i], lw=1, alpha=0.7)
        ax.grid(True, linewidth=0.3); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}", fontsize=9); ax.tick_params(labelsize=7)
        ax.set_box_aspect(1); ax.set_aspect("equal", adjustable="datalim")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3)); ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if created:
        ax3d.get_figure().subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.98)
    return [ax3d, *ax2d]
