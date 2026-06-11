"""
Report generation for the multi-approach fMCG pipeline.

Each save_* function is self-contained: it takes the data it needs and writes
one or more PDF/PNG files to output_dir. Errors are caught and logged so that
a failed plot never aborts the pipeline.
"""

import logging
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fmcg.analysis.heartbeat_averaging import segment_spans_seconds
from fmcg.utils.plotting.report import report_unified_hr_baseline
from fmcg.utils.utils import as_2d

logger = logging.getLogger(__name__)


PIPELINE_REPORT_KEYS = {
    "hr_baseline",
    "hr_per_method",
    "averaged_beats",
    "topk_averages",
    "average_comparison",
    "excerpt_comparison",
    "field_segments",
    "field_channels",
    "dipole_moments_segments",
    "dipole_moments_channels",
}

CONTROLLED_COMPARISON_REPORT_KEYS = {
    "hr_baseline",
    "topk_averages",
    "average_comparison",
    "field_segments",
    "field_channels",
    "dipole_moments_segments",
    "dipole_moments_channels",
    "excerpt_comparison",
}


def normalize_report_keys(
    reports: Optional[Any],
    valid_keys: Iterable[str],
    *,
    settings_enabled_key: str = "enabled",
    logger_: Optional[logging.Logger] = None,
) -> Set[str]:
    """Normalize report selection into a validated key set."""
    allowed = {
        str(key).strip().lower().replace("-", "_")
        for key in valid_keys
        if key is not None and str(key).strip()
    }

    if isinstance(reports, dict):
        reports = reports.get(settings_enabled_key, True)

    if isinstance(reports, bool):
        return set(allowed) if reports else set()

    if not reports:
        return set()

    report_keys = [reports] if isinstance(reports, str) else reports
    normalized: Set[str] = set()
    for key in report_keys:
        if key is None:
            continue
        cleaned = str(key).strip().lower().replace("-", "_")
        if cleaned:
            normalized.add(cleaned)

    unknown = sorted(normalized - allowed)
    if unknown:
        active_logger = logger_ if logger_ is not None else logger
        active_logger.warning("Unknown report keys ignored: %s", ", ".join(unknown))

    return normalized & allowed


def _select_field_signal(result, n_samples: int) -> Optional[np.ndarray]:
    domains = result.domains

    if result.name == "forward_model" and isinstance(domains.get("field"), np.ndarray):
        return as_2d(domains["field"])
    if isinstance(domains.get("field_fetal"), np.ndarray):
        return as_2d(domains["field_fetal"])
    if isinstance(domains.get("field"), np.ndarray):
        return as_2d(domains["field"])

    arrays = [v for v in domains.values() if isinstance(v, np.ndarray)]
    for arr in arrays:
        if arr.shape[0] == n_samples and arr.ndim >= 2:
            return as_2d(arr)

    return None


def _extract_shared_reference(approach_results, config, preprocessed):
    output_cfg = config.get("output", {})
    primary = output_cfg.get("primary_approach", "forward_model")
    preferred = output_cfg.get("report_reference_approach", "ica")

    ordered = []
    for name in (preferred, "ica", primary, *approach_results.keys()):
        if name not in ordered:
            ordered.append(name)

    for name in ordered:
        result = approach_results.get(name)
        if result is None:
            continue
        pp = result.metadata.get("postprocessing") or {}
        fetal = pp.get("fetal")
        if not fetal:
            continue

        peaks = np.asarray(fetal.get("peaks", []))
        if peaks.size < 2:
            continue

        outlier = fetal.get("outlier_mask")
        if outlier is not None and len(outlier) == len(peaks):
            peaks_clean = peaks[~np.asarray(outlier, dtype=bool)]
        else:
            peaks_clean = peaks

        hr_data = fetal.get("hr_data") or {}
        heart_rate = np.asarray(hr_data.get("heart_rate", []))
        baseline_hr = hr_data.get("baseline_hr")
        if baseline_hr is None and heart_rate.size > 0:
            baseline_hr = float(np.nanmean(heart_rate))

        return {
            "source": name,
            "peaks": peaks_clean,
            "segments": fetal.get("segments") or [],
            "heart_rate": heart_rate,
            "baseline_hr": baseline_hr,
            "bpm_threshold": hr_data.get("bpm_threshold", 5.0),
            "vss": hr_data.get("vss"),
            "time_axis": hr_data.get("time_axis", preprocessed.time),
            "detections": hr_data.get("detections"),
            "fs": preprocessed.fs,
        }
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_reports(approach_results, preprocessed, output_dir, config):
    """
    Generate all enabled reports after post-processing.

    Called from MultiApproachPipeline._post_process().
    """
    os.makedirs(output_dir, exist_ok=True)

    output_cfg = config.get("output", {})
    report_selection = output_cfg.get("reports", output_cfg.get("save_reports", True))
    enabled = normalize_report_keys(
        report_selection,
        PIPELINE_REPORT_KEYS,
        logger_=logger,
    )
    if not enabled:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    top_k = int(config.get("postprocessing", {}).get("top_k_segments", 3))
    snippet_start = float(output_cfg.get("comparison_snippet_start_seconds", 30.0))
    snippet_seconds = float(output_cfg.get("comparison_snippet_seconds", 5.0))
    channel_start = output_cfg.get("field_channel_report_start_seconds", 30.0)
    channel_end = output_cfg.get("field_channel_report_end_seconds", 60.0)
    field_max_rows = int(output_cfg.get("field_report_max_rows_per_page", 15))
    field_row_hspace = float(output_cfg.get("field_report_row_hspace", 0.05))

    shared_reference = _extract_shared_reference(approach_results, config, preprocessed)

    if "hr_baseline" in enabled and shared_reference is not None:
        report_unified_hr_baseline(
            shared_reference,
            plots_dir,
            report_name="10_hr_baseline_unified_REPORT.pdf",
        )

    for name, result in approach_results.items():
        if result is None:
            continue
        pp = result.metadata.get("postprocessing") or {}

        if "hr_per_method" in enabled:
            for entity in ("fetal", "maternal"):
                if entity in pp:
                    save_hr_report(
                        pp[entity],
                        preprocessed.fs,
                        label=entity,
                        savename=os.path.join(plots_dir, f"{name}_{entity}_hr_report.pdf"),
                    )

        if "averaged_beats" in enabled:
            save_averaged_beats_report(result, preprocessed, plots_dir, approach_name=name)

        if "topk_averages" in enabled:
            save_topk_segment_reports(result, plots_dir, approach_name=name, top_k=top_k)

    if "average_comparison" in enabled:
        save_topk_average_comparison_reports(approach_results, plots_dir, top_k=top_k)

    if "excerpt_comparison" in enabled:
        save_excerpt_comparison_report(
            approach_results,
            preprocessed,
            plots_dir,
            start_seconds=snippet_start,
            seconds=snippet_seconds,
            shared_reference=shared_reference,
        )

    if any(
        key in enabled
        for key in (
            "field_segments",
            "field_channels",
            "dipole_moments_segments",
            "dipole_moments_channels",
        )
    ):
        save_field_and_dipole_reports(
            approach_results,
            preprocessed,
            plots_dir,
            enabled=enabled,
            shared_reference=shared_reference,
            channel_start=channel_start,
            channel_end=channel_end,
            field_max_rows=field_max_rows,
            field_row_hspace=field_row_hspace,
        )


# ---------------------------------------------------------------------------
# HR report (HR trace + actogram + histogram)
# ---------------------------------------------------------------------------

def save_hr_report(postprocessing_entity, fs, label, savename):
    """
    Save the unified HR / actogram / histogram report for one entity.

    postprocessing_entity : dict
        Entry from result.metadata["postprocessing"]["fetal"|"maternal"].
        Must have "hr_data" key; "segments" is optional.
    """
    from fmcg.analysis.heartbeat_averaging import plot_unified_fetal_analysis

    hr_data = postprocessing_entity.get("hr_data")
    if not hr_data or "heart_rate" not in hr_data:
        logger.debug(f"HR report for {label}: no hr_data, skipping")
        return

    segments = postprocessing_entity.get("segments") or []

    # plot_unified_fetal_analysis needs the full hr_data keys; fall back gracefully
    required = {"heart_rate", "baseline_hr", "bpm_threshold", "vss", "time_axis", "detections"}
    if not required.issubset(hr_data.keys()):
        logger.debug(f"HR report for {label}: incomplete hr_data keys, skipping")
        return

    try:
        fig = plot_unified_fetal_analysis(segments, hr_data)
        fig.savefig(savename, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  HR report ({label}) → {savename}")
    except Exception as e:
        logger.warning(f"  HR report ({label}) failed: {e}")


# ---------------------------------------------------------------------------
# Averaged beats report
# ---------------------------------------------------------------------------

def save_averaged_beats_report(result, preprocessed, output_dir, approach_name):
    """
    Save averaged beat comparison plot for all domains of one approach.

    For multi-channel domains (field reconstruction), groups channels by spatial
    axis (x/y/z) using axis_mask and plots the mean per axis — giving 3 meaningful
    traces instead of a single L2-norm collapse.
    """
    from fmcg.utils.plotting.plot_utils import plot_averaged_beats_comparison

    pp = result.metadata.get("postprocessing") or {}

    # Precompute axis grouping for multi-channel domains
    axis_mask = preprocessed.axis_mask  # (S, 3)
    S = axis_mask.shape[0]
    axis_indices = np.tile(np.arange(3), S).reshape(-1, 3)[axis_mask.astype(bool)]  # (n_valid,)

    def _to_axis_means(arr):
        """Reduce (T, n_valid) to (T, 3) by averaging channels per axis."""
        out = np.zeros((arr.shape[0], 3))
        for i in range(3):
            mask = axis_indices == i
            if mask.any():
                out[:, i] = arr[:, mask].mean(axis=1)
        return out

    for entity in ("fetal", "maternal"):
        if entity not in pp:
            continue
        beat_averages = pp[entity].get("beat_averages", {})
        if not beat_averages:
            continue

        averaged = []
        stds = []
        labels = []
        time_axis = None

        for domain_name, avg in beat_averages.items():
            mean_e = avg["mean"]  # (T_epoch, n_channels)
            std_e  = avg["std"]
            if time_axis is None:
                time_axis = avg["time"]

            n_ch = mean_e.shape[1]
            if n_ch == 3:
                # Already per-axis (forward model dipole domain)
                averaged.append(mean_e)
                stds.append(std_e)
            elif n_ch == len(axis_indices):
                # Multi-channel field: reduce to per-axis mean
                averaged.append(_to_axis_means(mean_e))
                stds.append(_to_axis_means(std_e))
            else:
                # Unknown shape: fall back to L2-norm
                averaged.append(np.linalg.norm(mean_e, axis=1, keepdims=True))
                stds.append(np.linalg.norm(std_e, axis=1, keepdims=True))
            labels.append(domain_name)

        if not averaged:
            continue

        savename = os.path.join(
            output_dir, f"{approach_name}_{entity}_averaged_beats.pdf"
        )
        try:
            plot_averaged_beats_comparison(
                averaged,
                std_beats=stds,
                method_labels=labels,
                time=time_axis,
                include_magnitude=False,
                savename=savename,
            )
            logger.info(f"  Averaged beats ({approach_name}/{entity}) → {savename}")
        except Exception as e:
            logger.warning(f"  Averaged beats ({approach_name}/{entity}) failed: {e}")


def save_topk_segment_reports(result, output_dir, approach_name, top_k=3):
    """Save top-k segment reports from additive postprocessing metadata."""
    from fmcg.utils.plotting.report import report_topk_averages

    pp = result.metadata.get("postprocessing") or {}
    for entity in ("fetal", "maternal"):
        beat_averages = pp.get(entity, {}).get("beat_averages", {})
        for domain_name, domain_avg in beat_averages.items():
            seg_entries = domain_avg.get("segments") or []
            payload = []
            if seg_entries:
                for seg in seg_entries[: max(1, int(top_k))]:
                    payload.append(
                        {
                            "mean": np.asarray(seg["mean"]),
                            "std": np.asarray(seg["std"]),
                            "time_axis": np.asarray(seg["time"]),
                            "segment": {
                                "count": seg.get("beat_count"),
                                "start_sec": seg.get("start_sec"),
                                "end_sec": seg.get("end_sec"),
                                "mean_hr": seg.get("mean_hr"),
                            },
                        }
                    )
            else:
                payload.append(
                    {
                        "mean": np.asarray(domain_avg["mean"]),
                        "std": np.asarray(domain_avg["std"]),
                        "time_axis": np.asarray(domain_avg["time"]),
                        "segment": None,
                    }
                )

            if not payload:
                continue

            report_topk_averages(
                method_name=f"{approach_name}:{entity}:{domain_name}",
                averages=payload,
                output_dir=output_dir,
                report_name=f"{approach_name}_{entity}_{domain_name}_topk_averages_REPORT.pdf",
                ncol=2,
                top_k=top_k,
            )


def save_topk_average_comparison_reports(approach_results, output_dir, top_k=3):
    """Save cross-method top-k average comparison reports grouped by entity/domain."""
    from fmcg.utils.plotting.report import report_topk_average_comparison

    grouped: Dict[Tuple[str, str], Dict[str, Sequence[Dict[str, Any]]]] = {}
    fallback_by_entity: Dict[str, Dict[str, Sequence[Dict[str, Any]]]] = {"fetal": {}, "maternal": {}}
    for approach_name, result in approach_results.items():
        if result is None:
            continue
        pp = result.metadata.get("postprocessing") or {}
        for entity in ("fetal", "maternal"):
            beat_averages = pp.get(entity, {}).get("beat_averages", {})
            if not beat_averages:
                continue

            first_domain = next(iter(beat_averages.values()))
            first_segments = first_domain.get("segments") or []
            if first_segments:
                fallback_entries = [
                    {
                        "mean": np.asarray(seg["mean"]),
                        "std": np.asarray(seg["std"]),
                        "time_axis": np.asarray(seg["time"]),
                        "segment": {
                            "count": seg.get("beat_count"),
                            "start_sec": seg.get("start_sec"),
                            "end_sec": seg.get("end_sec"),
                            "mean_hr": seg.get("mean_hr"),
                        },
                    }
                    for seg in first_segments
                ]
            else:
                fallback_entries = [
                    {
                        "mean": np.asarray(first_domain["mean"]),
                        "std": np.asarray(first_domain["std"]),
                        "time_axis": np.asarray(first_domain["time"]),
                        "segment": None,
                    }
                ]
            fallback_by_entity[entity][approach_name] = fallback_entries

            for domain_name, avg in beat_averages.items():
                seg_entries = avg.get("segments") or []
                if seg_entries:
                    entries = []
                    for seg in seg_entries:
                        entries.append(
                            {
                                "mean": np.asarray(seg["mean"]),
                                "std": np.asarray(seg["std"]),
                                "time_axis": np.asarray(seg["time"]),
                                "segment": {
                                    "count": seg.get("beat_count"),
                                    "start_sec": seg.get("start_sec"),
                                    "end_sec": seg.get("end_sec"),
                                    "mean_hr": seg.get("mean_hr"),
                                },
                            }
                        )
                else:
                    entries = [
                        {
                            "mean": np.asarray(avg["mean"]),
                            "std": np.asarray(avg["std"]),
                            "time_axis": np.asarray(avg["time"]),
                            "segment": None,
                        }
                    ]

                grouped.setdefault((entity, domain_name), {})[approach_name] = entries

    wrote_any = False
    for (entity, domain_name), methods in grouped.items():
        if len(methods) < 2:
            continue
        safe_domain = domain_name.replace("/", "_").replace(":", "_")
        report_topk_average_comparison(
            methods,
            output_dir,
            report_name=f"{entity}_{safe_domain}_average_comparison_REPORT.pdf",
            ncol=2,
            top_k=top_k,
        )
        wrote_any = True

    if not wrote_any:
        for entity in ("fetal", "maternal"):
            methods = fallback_by_entity.get(entity, {})
            if len(methods) < 2:
                continue
            report_topk_average_comparison(
                methods,
                output_dir,
                report_name=f"{entity}_primary_average_comparison_REPORT.pdf",
                ncol=2,
                top_k=top_k,
            )


def save_excerpt_comparison_report(
    approach_results,
    preprocessed,
    output_dir,
    start_seconds=30.0,
    seconds=5.0,
    shared_reference=None,
):
    """Save multi-method unaveraged field excerpt comparison plot."""
    from fmcg.utils.plotting.report import report_excerpt_comparison

    methods: Dict[str, np.ndarray] = {}

    n_samples = len(preprocessed.time)
    for name, result in approach_results.items():
        if result is None:
            continue
        sig = _select_field_signal(result, n_samples=n_samples)
        if sig is None or sig.shape[0] < 2:
            continue
        methods[name] = sig

    if len(methods) < 2:
        return

    fs = float(preprocessed.fs)
    spans = []
    if shared_reference is not None:
        n = max(1, int(seconds * fs))
        max_start = max(0, min(sig.shape[0] for sig in methods.values()) - n)
        start_idx = int(round(start_seconds * fs))
        start_idx = max(0, min(start_idx, max_start))
        end_idx = start_idx + n
        time_excerpt = np.arange(start_idx, end_idx) / fs
        spans = segment_spans_seconds(
            np.asarray(shared_reference.get("peaks", [])),
            shared_reference.get("segments", []),
            fs,
            min_time=float(time_excerpt[0]),
            max_time=float(time_excerpt[-1]),
        )

    report_excerpt_comparison(
        methods,
        fs,
        output_dir,
        report_name="excerpt_comparison_REPORT.pdf",
        start_seconds=start_seconds,
        seconds=seconds,
        selected_spans=spans,
        title="Method Comparison: Unaveraged Field Excerpts",
    )


def save_field_and_dipole_reports(
    approach_results,
    preprocessed,
    output_dir,
    enabled,
    shared_reference=None,
    channel_start=30.0,
    channel_end=60.0,
    field_max_rows=15,
    field_row_hspace=0.05,
):
    """Save Holter-like field and dipole segment reports."""
    from fmcg.utils.plotting.report import (
        report_field_segments,
        report_field_segments_per_channel,
        report_m_hat_segments,
        report_m_hat_segments_per_channel,
    )

    fs = float(preprocessed.fs)
    time_full = preprocessed.time

    spans = []
    if shared_reference is not None:
        spans = segment_spans_seconds(
            np.asarray(shared_reference.get("peaks", [])),
            shared_reference.get("segments", []),
            fs,
            min_time=float(time_full[0]),
            max_time=float(time_full[-1]),
        )

    if "field_segments" in enabled or "field_channels" in enabled:
        for name, result in approach_results.items():
            if result is None:
                continue
            sig = _select_field_signal(result, n_samples=len(time_full))
            if sig is None:
                continue

            payload = {name: {"field": sig, "selected_spans": spans}}
            if "field_segments" in enabled:
                report_field_segments(
                    payload,
                    time_full,
                    output_dir,
                    segment_duration=5,
                    report_name=f"{name}_field_segments_REPORT.pdf",
                )
            if "field_channels" in enabled:
                start_label = "full" if channel_start is None else f"{float(channel_start):.0f}s"
                end_label = "end" if channel_end is None else f"{float(channel_end):.0f}s"
                report_field_segments_per_channel(
                    payload,
                    time_full,
                    output_dir,
                    segment_duration=5,
                    start_seconds=channel_start,
                    end_seconds=channel_end,
                    report_name=f"{name}_field_channels_{start_label}_{end_label}_REPORT.pdf",
                    max_rows_per_page=field_max_rows,
                    row_hspace=field_row_hspace,
                )

    forward = approach_results.get("forward_model")
    if forward is not None and (
        "dipole_moments_segments" in enabled or "dipole_moments_channels" in enabled
    ):
        dipole_entries = {}
        dipole = forward.domains.get("dipole_fetal")
        maternal = forward.domains.get("dipole_maternal")

        if isinstance(dipole, np.ndarray):
            dipole_entries["fetal"] = {"dipole_moments": as_2d(dipole)}
        if isinstance(maternal, np.ndarray):
            dipole_entries["maternal"] = {"dipole_moments": as_2d(maternal)}

        if dipole_entries:
            if "dipole_moments_segments" in enabled:
                report_m_hat_segments(
                    dipole_entries,
                    time_full,
                    output_dir,
                    segment_duration=5,
                    report_name="forward_model_dipole_moments_segments_REPORT.pdf",
                )
            if "dipole_moments_channels" in enabled:
                start_label = "full" if channel_start is None else f"{float(channel_start):.0f}s"
                end_label = "end" if channel_end is None else f"{float(channel_end):.0f}s"
                report_m_hat_segments_per_channel(
                    dipole_entries,
                    time_full,
                    output_dir,
                    segment_duration=5,
                    start_seconds=channel_start,
                    end_seconds=channel_end,
                    report_name=f"forward_model_dipole_moments_channels_{start_label}_{end_label}_REPORT.pdf",
                    max_rows_per_page=6,
                    row_hspace=0.15,
                )

def save_legacy_reports(data_dict, heartbeats_dict, time_, output_dir):
    """
    Generate legacy-style segmented time-trace PDFs using existing report functions.

    Delegates to fmcg.utils.plotting.report for dipole moment and position traces.
    HR reporting is handled by save_hr_report() (new-style) since the new pipeline
    stores per-beat HR rather than timeline-indexed HR that report_hr() expects.

    Produces:
      - 2_dipole_moments_REPORT.pdf   (dipole moment time trace, forward model only)
      - 2_dipole_positions_REPORT.pdf (dipole position time trace, forward model only)
    """
    from fmcg.utils.plotting.report import (
        report_m_hat_segments,
        report_r_hat_segments,
    )

    if not data_dict:
        return

    # Build m_hat / r_hat dicts from data_dict for dipole-based entries
    m_hat_dict = {
        k: {"dipole_moments": v["dipole_moments"]}
        for k, v in data_dict.items()
        if "dipole_moments" in v
    }
    if m_hat_dict:
        try:
            report_m_hat_segments(m_hat_dict, time_, output_dir)
        except Exception as e:
            logger.warning(f"  Legacy dipole moment report failed: {e}")

    r_hat_dict = {
        k: {"position": v["position"]}
        for k, v in data_dict.items()
        if "position" in v
    }
    if r_hat_dict:
        try:
            report_r_hat_segments(r_hat_dict, time_, output_dir)
        except Exception as e:
            logger.warning(f"  Legacy dipole position report failed: {e}")

