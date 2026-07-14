"""Phase-warped, cluster-averaged heartbeat template (Oster et al. 2015).

Plain ensemble averaging aligns beats by sample offset and averages every beat,
which smears morphology under HR variation and lets ectopic or misdetected beats
corrupt the template. Instead we warp each beat to a common cardiac-phase grid
(QRS at -pi/3, 250 bins), cluster beats by QRS cross-correlation, and average
only the largest cluster.

Correlation is measured on a QRS-centred window (corr_halfwidth_ms), not the
whole cycle, since over a full cycle the QRS is a small fraction of the samples
and the T-wave/baseline noise dominates the similarity. Beats are baseline
detrended first. Correlation is summed over channels so one beat set is chosen
for the whole VCG.

    from fmcg.analysis.phase_template import build_phase_template
"""

import numpy as np


def resample_beats_by_phase(signal, peaks, fs, n_bins=250, qrs_phase=-np.pi / 3):
    """Warp each beat onto a common cardiac-phase grid.

    Each beat spans one cycle [R - f_pre*RR, R + (1-f_pre)*RR] with the R-peak at
    qrs_phase, resampled to n_bins. RR is the mean of the two adjacent inter-peak
    intervals, so the warp follows HR variation.

    signal: (n_samples,) or (n_samples, n_channels); peaks: R-peak sample indices.
    Returns beats (n_valid, n_bins, n_channels), phase_axis (n_bins,) in [-pi, pi],
    and the sample indices of the peaks that produced a beat.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, None]
    n_samples, n_channels = signal.shape

    peaks = np.asarray(peaks)
    phase_axis = np.linspace(-np.pi, np.pi, n_bins)
    if peaks.size < 2:
        return np.empty((0, n_bins, n_channels)), phase_axis, np.empty(0, dtype=int)

    f_pre = (qrs_phase + np.pi) / (2 * np.pi)  # fraction of the cycle before R
    global_rr = float(np.diff(peaks).mean())
    xp = np.arange(n_samples)

    beats, kept_peaks = [], []
    for i, r in enumerate(peaks):
        prev_rr = r - peaks[i - 1] if i > 0 else global_rr
        next_rr = peaks[i + 1] - r if i < len(peaks) - 1 else global_rr
        rr = 0.5 * (prev_rr + next_rr)

        win_start = r - f_pre * rr
        win_end = r + (1 - f_pre) * rr
        if win_start < 0 or win_end > n_samples - 1:
            continue

        query = np.linspace(win_start, win_end, n_bins)
        beat = np.stack(
            [np.interp(query, xp, signal[:, c]) for c in range(n_channels)], axis=1
        )
        if np.isnan(beat).any():
            continue
        beats.append(beat)
        kept_peaks.append(r)

    if not beats:
        return np.empty((0, n_bins, n_channels)), phase_axis, np.empty(0, dtype=int)
    return np.stack(beats), phase_axis, np.asarray(kept_peaks, dtype=int)


def _detrend_beats(beats):
    """Subtract a per-beat, per-channel linear baseline."""
    x = np.linspace(0.0, 1.0, beats.shape[1])
    A = np.vstack([x, np.ones_like(x)]).T
    proj = A @ np.linalg.pinv(A)
    return beats - np.einsum("ij,njc->nic", proj, beats)


def _normalise(seg):
    """Mean-remove and unit-normalise each flattened beat window."""
    seg = seg - seg.mean(axis=1, keepdims=True)
    return seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)


def _grow_cluster(seg, pool, seed, theta):
    """Template-matching: grow from seed, keep pool beats with corr >= theta, iterate."""
    template = seg[seed].copy()
    members = np.array([seed])
    for _ in range(10):
        members_new = pool[np.flatnonzero(seg[pool] @ template >= theta)]
        if members_new.size == 0:
            break
        template_new = _normalise(seg[members_new].mean(axis=0)[None])[0]
        converged = np.dot(template_new, template) > 0.9999
        template, members = template_new, members_new
        if converged:
            break
    return members


def select_cluster(
    beats, qrs_slice=None, min_members=30, thr_start=0.90, thr_step=0.05, thr_min=0.60
):
    """Largest QRS-correlation cluster, seeded from the most representative beat.

    Sweeps the threshold from thr_start down to thr_min and returns the first
    cluster with at least min_members beats; if none qualifies, returns the
    largest found. Correlation uses the flattened QRS window across all channels.
    Returns member indices, threshold used, and whether min_members was reached.
    """
    n = beats.shape[0]
    if n == 0:
        return np.empty(0, dtype=int), np.nan, False

    window = beats[:, qrs_slice, :] if qrs_slice is not None else beats
    seg = _normalise(window.reshape(n, -1))
    pool = np.arange(n)

    sim = seg @ seg.T
    np.fill_diagonal(sim, 0.0)
    seed = int(sim.sum(axis=1).argmax())

    best_members, best_threshold, best_size = pool, thr_min, -1
    for theta in np.arange(thr_start, thr_min - 1e-9, -thr_step):
        members = _grow_cluster(seg, pool, seed, theta)
        if members.size > best_size:
            best_size, best_members, best_threshold = (
                members.size,
                members,
                float(theta),
            )
        if members.size >= min_members:
            return members, float(theta), True
    return best_members, best_threshold, False


def select_secondary_cluster(
    beats,
    primary_idx,
    qrs_slice=None,
    min_members=5,
    thr_start=0.90,
    thr_step=0.05,
    thr_min=0.50,
):
    """Secondary (e.g. ectopic) mode distinct from the primary cluster.

    Unlike select_cluster, the seed is the non-primary beat least correlated with
    the primary template, so the cluster locks onto the morphology that differs
    most from the primary rather than the largest near-primary variant. Same
    return signature as select_cluster.
    """
    n = beats.shape[0]
    pool = np.setdiff1d(np.arange(n), np.asarray(primary_idx))
    if pool.size == 0:
        return np.empty(0, dtype=int), np.nan, False

    window = beats[:, qrs_slice, :] if qrs_slice is not None else beats
    seg = _normalise(window.reshape(n, -1))
    primary_template = _normalise(seg[primary_idx].mean(axis=0)[None])[0]
    seed = pool[np.argmin(seg[pool] @ primary_template)]

    best_members, best_threshold, best_size = pool[:1], thr_min, -1
    for theta in np.arange(thr_start, thr_min - 1e-9, -thr_step):
        members = _grow_cluster(seg, pool, seed, theta)
        if members.size > best_size:
            best_size, best_members, best_threshold = (
                members.size,
                members,
                float(theta),
            )
        if members.size >= min_members:
            return members, float(theta), True
    return best_members, best_threshold, False


def build_phase_template(
    signal,
    peaks,
    fs,
    n_bins=250,
    qrs_phase=-np.pi / 3,
    corr_halfwidth_ms=60.0,
    detrend=True,
    min_members=30,
    thr_start=0.90,
    thr_step=0.05,
    thr_min=0.60,
    plot=False,
    title="",
):
    """Phase-warped, cluster-averaged heartbeat template.

    Detect peaks on the magnitude / dominant channel for a clean set: false
    positives break the clustering. corr_halfwidth_ms sets the QRS window used for
    the clustering correlation; the template average still spans the whole cycle.

    Returns a dict with: mean, std (n_bins, n_channels); phase_axis; time_axis
    (s, relative to the R-peak via median RR); qrs_slice; member_idx; kept_peaks;
    beats (phase-warped, detrended); n_members; n_total; threshold_used; relevant.
    """
    beats, phase_axis, kept_peaks = resample_beats_by_phase(
        signal, peaks, fs, n_bins=n_bins, qrs_phase=qrs_phase
    )
    if detrend and beats.size:
        beats = _detrend_beats(beats)

    peaks = np.asarray(peaks)
    rr_sec = (float(np.median(np.diff(peaks))) if peaks.size >= 2 else n_bins) / fs
    time_axis = (phase_axis - qrs_phase) / (2 * np.pi) * rr_sec  # t=0 at the R-peak

    qc = int(round((qrs_phase + np.pi) / (2 * np.pi) * (n_bins - 1)))
    half_bins = int(round((corr_halfwidth_ms / 1000.0) / rr_sec * n_bins))
    qrs_slice = slice(max(0, qc - half_bins), min(n_bins, qc + half_bins + 1))

    n_channels = (
        beats.shape[2]
        if beats.size
        else (signal.shape[1] if np.ndim(signal) > 1 else 1)
    )

    if beats.shape[0] == 0:
        nan = np.full((n_bins, n_channels), np.nan)
        return dict(
            mean=nan,
            std=nan.copy(),
            phase_axis=phase_axis,
            time_axis=time_axis,
            qrs_slice=qrs_slice,
            member_idx=np.empty(0, dtype=int),
            kept_peaks=kept_peaks,
            beats=beats,
            n_members=0,
            n_total=0,
            threshold_used=np.nan,
            relevant=False,
        )

    member_idx, threshold_used, relevant = select_cluster(
        beats,
        qrs_slice=qrs_slice,
        min_members=min_members,
        thr_start=thr_start,
        thr_step=thr_step,
        thr_min=thr_min,
    )
    cluster = beats[member_idx]

    if plot:
        from .heartbeat_averaging import _plot_average_heartbeat

        _plot_average_heartbeat(
            cluster.mean(axis=0), cluster.std(axis=0), time_axis, title=title
        )

    return dict(
        mean=cluster.mean(axis=0),
        std=cluster.std(axis=0),
        phase_axis=phase_axis,
        time_axis=time_axis,
        qrs_slice=qrs_slice,
        member_idx=member_idx,
        kept_peaks=kept_peaks,
        beats=beats,
        n_members=int(member_idx.size),
        n_total=int(beats.shape[0]),
        threshold_used=threshold_used,
        relevant=relevant,
    )
