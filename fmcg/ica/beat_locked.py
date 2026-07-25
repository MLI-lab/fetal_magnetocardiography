"""Beat-locked component test for ICA annotation.

Locks an ICA source to a reference R-peak train and measures how consistently it
carries that beat. A fetal component gives a clean averaged beat under the fetal
reference and a flat one under the maternal reference; a maternal component shows
the reverse. Used by ICAComponentLabeler to support annotation, and usable on its
own. Minimal and self-contained.
"""
import numpy as np

from fmcg.analysis.heartbeat_averaging import extract_epochs


def beat_locked_average(source, peaks, fs, ratio_pre=0.5, interval=1.5):
    """Average and spread of a source aligned to a reference peak train.

    Returns (mean, std, n_beats): mean and std are 1D over the epoch window,
    n_beats is the number of aligned epochs. Returns (None, None, 0) if there are
    too few peaks to average.
    """
    peaks = np.asarray(peaks)
    if peaks.size < 4:
        return None, None, 0
    epochs, _ = extract_epochs(np.asarray(source)[:, None], peaks, fs, ratio_pre, interval)
    if epochs.shape[0] < 2:
        return None, None, int(epochs.shape[0])
    ep = epochs[:, :, 0]
    return ep.mean(0), ep.std(0, ddof=1), int(ep.shape[0])


def f_statistic(mean, std, n_beats):
    """Beat-lock F statistic: mean-square of the average beat over the mean-square
    of the across-beat standard deviation, scaled by the beat count. Large means
    tight locking. Returns nan when it cannot be formed.
    """
    if mean is None or n_beats < 2:
        return np.nan
    denom = float((std ** 2).mean())
    if denom <= 0:
        return np.nan
    return float((mean ** 2).mean() / denom * n_beats)


def beat_lock(source, peaks, fs, n_perm=0, alpha=0.01, ratio_pre=0.5,
              interval=1.5, rng=None):
    """Lock a source to a reference peak train and score the locking.

    Returns a dict with keys mean, std, n_beats, F, threshold, significant,
    p_value. With n_perm=0 the permutation null is skipped, so threshold,
    significant and p_value are None and only the average beat and the F
    statistic are returned. With n_perm>0 the F statistic is compared to a null
    built by shifting the peaks to random positions; significant is F above the
    (1-alpha) percentile of that null. The default rng is seeded, so the test is
    reproducible.
    """
    source = np.asarray(source)
    mean, std, n = beat_locked_average(source, peaks, fs, ratio_pre, interval)
    F = f_statistic(mean, std, n)
    out = dict(mean=mean, std=std, n_beats=n, F=F,
               threshold=None, significant=None, p_value=None)
    if n_perm <= 0 or not np.isfinite(F):
        return out

    if rng is None:
        rng = np.random.default_rng(0)
    n_total = len(source)
    peaks = np.sort(np.asarray(peaks))
    guard = int(fs)
    if n_total <= 2 * guard:
        return out
    null = []
    for _ in range(int(n_perm)):
        shift = int(rng.integers(guard, n_total - guard))
        ps = peaks + shift
        ps = ps[(ps >= 0) & (ps < n_total)]
        if ps.size <= 3:
            continue
        m, s, k = beat_locked_average(source, np.unique(ps), fs, ratio_pre, interval)
        Fp = f_statistic(m, s, k)
        if np.isfinite(Fp):
            null.append(Fp)
    if not null:
        return out
    null = np.asarray(null)
    out["threshold"] = float(np.percentile(null, 100 * (1 - alpha)))
    out["p_value"] = float(np.mean(null >= F))
    out["significant"] = bool(F > out["threshold"])
    return out
