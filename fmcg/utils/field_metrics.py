"""
Field-level evaluation metrics for fMCG reconstruction.

Migrated from experiments/evaluation/synthetic_evaluation/annotate_synthetic_ica.py
so that notebooks and experiments can share the same implementations.
"""

import numpy as np


def _pearson_r(x, y):
    x, y = x.flatten(), y.flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _fit_best_scale(pred, ref):
    """Scalar s minimising ||s·pred - ref||²  (closed-form least squares)."""
    denom = float(pred @ pred)
    return float(pred @ ref) / denom if denom > 0 else np.nan


def compute_field_metrics(field_pred, field_ref):
    """
    Compute reconstruction metrics between predicted and reference fields.

    Both inputs must be in the same domain and units (e.g. both physical [pT]
    or both whitened).

    Parameters
    ----------
    field_pred, field_ref : array-like, shape (T, C) or (T,)
        Predicted and reference field arrays.

    Returns
    -------
    dict with keys:
        field_mae_pT, field_rmse_pT, field_rel_error_pct,
        field_rel_error_peak_pct, field_pearson_r, field_scale,
        field_rel_error_peak_pct_scaled, field_r2, field_num_valid_samples
    """
    default = dict(
        field_mae_pT=np.nan,
        field_rmse_pT=np.nan,
        field_rel_error_pct=np.nan,
        field_rel_error_peak_pct=np.nan,
        field_pearson_r=np.nan,
        field_scale=np.nan,
        field_rel_error_peak_pct_scaled=np.nan,
        field_r2=np.nan,
        field_num_valid_samples=0,
    )

    if field_pred is None or field_ref is None:
        return default

    pred = np.asarray(field_pred, dtype=float)
    ref = np.asarray(field_ref, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if ref.ndim == 1:
        ref = ref[:, None]

    n = min(pred.shape[0], ref.shape[0])
    pred, ref = pred[:n], ref[:n]

    valid = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(valid):
        return default

    pred_v = pred[valid]
    ref_v = ref[valid]
    diff_v = pred_v - ref_v

    mae = float(np.mean(np.abs(diff_v)))
    rmse = float(np.sqrt(np.mean(diff_v ** 2)))

    ref_norm = float(np.linalg.norm(ref_v))
    rel_error_pct = 100.0 * float(np.linalg.norm(diff_v)) / ref_norm if ref_norm > 0 else np.nan

    ref_peak = float(np.max(np.abs(ref_v))) if ref_v.size else np.nan
    rel_error_peak_pct = (
        100.0 * rmse / ref_peak if np.isfinite(ref_peak) and ref_peak > 0 else np.nan
    )

    ss_tot = float(np.sum((ref_v - ref_v.mean()) ** 2))
    r2 = 1.0 - float(np.sum(diff_v ** 2)) / ss_tot if ss_tot > 0 else np.nan

    pearson = _pearson_r(pred_v, ref_v)
    scale = _fit_best_scale(pred_v.flatten(), ref_v.flatten())

    if np.isfinite(scale):
        rmse_scaled = float(np.sqrt(np.mean((scale * pred_v - ref_v) ** 2)))
        rel_error_peak_pct_scaled = (
            100.0 * rmse_scaled / ref_peak
            if np.isfinite(ref_peak) and ref_peak > 0
            else np.nan
        )
    else:
        rel_error_peak_pct_scaled = np.nan

    return dict(
        field_mae_pT=mae,
        field_rmse_pT=rmse,
        field_rel_error_pct=rel_error_pct,
        field_rel_error_peak_pct=rel_error_peak_pct,
        field_pearson_r=pearson,
        field_scale=scale,
        field_rel_error_peak_pct_scaled=rel_error_peak_pct_scaled,
        field_r2=r2,
        field_num_valid_samples=int(ref_v.size),
    )


def compute_named_field_metrics(entity, field_pred, field_ref):
    """
    Same as :func:`compute_field_metrics`, but keys are prefixed with
    ``field_{entity}_`` (e.g. ``field_fetal_mae_pT``).

    Parameters
    ----------
    entity : str
        Source label, e.g. ``"fetal"`` or ``"maternal"``.
    field_pred, field_ref : array-like, shape (T, C)
    """
    base = compute_field_metrics(field_pred, field_ref)
    # strip the "field_" prefix from each key and re-add with entity inserted
    return {f"field_{entity}_{k[len('field_'):]}" : v for k, v in base.items()}
