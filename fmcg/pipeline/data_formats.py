"""
Data format builders for the multi-approach fMCG pipeline.

Two output formats:
  pipeline_results.pkl    — legacy format, compatible with wave_annotater.py
  pipeline_results_v2.pkl — hierarchical format for evaluation and comparison
"""

import datetime
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_legacy_format(approach_results, preprocessed, primary_approach, config):
    """
    Build a pipeline_results.pkl-compatible dict from post-processed results.

    heartbeats_dict structure (per entity, per axis):
        {"fetal": {"M_x": {"mean_beat": pd.Series, "std_beat": pd.Series,
                            "mean_position": pd.Series, "std_position": pd.Series,
                            "mean_beat_field": ndarray, "std_beat_field": ndarray,
                            "mean_beat_ica": ndarray,   "std_beat_ica": ndarray,
                            "peaks": ndarray, "selected_peaks_mask": ndarray},
                   "M_y": {...}, "M_z": {...}},
         "maternal": {...}}

    Parameters
    ----------
    approach_results : dict
        {approach_name: ApproachResult} as stored in pipeline.results.
    preprocessed : PreprocessedData
    primary_approach : str
        Which approach populates the main mean_beat/std_beat entries.
        Defaults to "forward_model".
    config : dict
        Full pipeline config.
    """
    primary = approach_results.get(primary_approach)
    if primary is None:
        logger.warning(
            f"build_legacy_format: primary approach '{primary_approach}' not in results. "
            f"Available: {list(approach_results.keys())}"
        )
        return {"data_dict": {}, "heartbeats_dict": {}}

    pp_primary = (primary.metadata.get("postprocessing") or {})
    axis_mask = preprocessed.axis_mask  # (S, 3)

    # Axis-index array for valid channels: tells which component (0/1/2) each valid channel is
    # Shape (n_valid,), values in {0, 1, 2}
    S = axis_mask.shape[0]
    axis_indices = np.tile(np.arange(3), S).reshape(-1, 3)[axis_mask.astype(bool)]  # (n_valid,)

    data_dict = {}
    heartbeats_dict = {}

    for entity in ("fetal", "maternal"):
        if entity not in pp_primary:
            continue

        entity_pp = pp_primary[entity]
        peaks = entity_pp.get("peaks", np.array([]))
        outlier = entity_pp.get("outlier_mask", np.zeros(len(peaks), dtype=bool))
        hr_data = entity_pp.get("hr_data", {})

        # Build data_dict entry
        data_dict[entity] = {
            "peaks": peaks,
            "outlier": outlier,
            "hr": hr_data.get("heart_rate", np.array([])),
        }

        # Forward model: include dipole moments + positions for legacy segmented reports
        dipole_key = f"dipole_{entity}"
        pos_key = f"position_{entity}"
        if dipole_key in primary.domains:
            data_dict[entity]["dipole_moments"] = primary.domains[dipole_key]
        if pos_key in primary.domains:
            data_dict[entity]["position"] = primary.domains[pos_key]

        beat_avgs = entity_pp.get("beat_averages", {})

        # selected_peaks_mask: True for peaks inside any near-baseline segment
        segments = entity_pp.get("segments", [])
        range_segs = [(peaks[s], peaks[e - 1]) for (_, s, e) in segments if e > s]
        peaks_mask = np.array(
            [any(lo <= p <= hi for lo, hi in range_segs) for p in peaks],
            dtype=bool,
        )

        # Build heartbeats_dict[entity]["M_x/y/z"]
        entity_beats = {}
        for i, label in enumerate(["x", "y", "z"]):
            key = f"M_{label}"
            ch_mask = axis_indices == i  # select valid channels for this axis

            entry = {
                "peaks": peaks,
                "selected_peaks_mask": peaks_mask,
            }

            # --- Primary approach: mean/std beat (dipole moment or field magnitude) ---
            dipole_key = f"dipole_{entity}"
            field_key  = "field"

            if primary_approach == "forward_model" and dipole_key in beat_avgs:
                avg = beat_avgs[dipole_key]
                time_a = avg["time"]
                entry["mean_beat"] = pd.Series(avg["mean"][:, i], index=time_a)
                entry["std_beat"]  = pd.Series(avg["std"][:, i],  index=time_a)

                # Position averages
                pos_key = f"position_{entity}"
                if pos_key in beat_avgs:
                    p_avg = beat_avgs[pos_key]
                    entry["mean_position"] = pd.Series(p_avg["mean"][:, i], index=time_a)
                    entry["std_position"]  = pd.Series(p_avg["std"][:, i],  index=time_a)

                # Field averages per axis (full S*3 → select valid axis-i channels)
                if field_key in beat_avgs:
                    f_avg = beat_avgs[field_key]
                    # field was stored as (T, S, 3) reshaped to (T, S*3)
                    # We need a (S*3,) mask for valid channels at axis i
                    full_axis = np.tile(np.arange(3), S)     # (S*3,)
                    full_valid = axis_mask.flatten().astype(bool)  # (S*3,)
                    field_ch_mask = full_valid & (full_axis == i)  # (S*3,)
                    entry["mean_beat_field"] = f_avg["mean"][:, field_ch_mask]
                    entry["std_beat_field"]  = f_avg["std"][:, field_ch_mask]

            elif primary_approach == "ica":
                fetal_field_key = "field_fetal"
                if fetal_field_key in beat_avgs:
                    avg = beat_avgs[fetal_field_key]
                    time_a = avg["time"]
                    # field_fetal is (T, n_valid) — select channels for axis i
                    entry["mean_beat"] = pd.Series(
                        np.linalg.norm(avg["mean"][:, ch_mask], axis=1), index=time_a
                    )
                    entry["std_beat"] = pd.Series(
                        np.linalg.norm(avg["std"][:, ch_mask], axis=1), index=time_a
                    )
                    entry["mean_beat_field"] = avg["mean"][:, ch_mask]
                    entry["std_beat_field"]  = avg["std"][:, ch_mask]

            # --- ICA field averages (if ICA ran and primary is not ICA) ---
            ica_result = approach_results.get("ica")
            if ica_result is not None and primary_approach != "ica":
                ica_pp = (ica_result.metadata.get("postprocessing") or {})
                if entity in ica_pp:
                    ica_beat_avgs = ica_pp[entity].get("beat_averages", {})
                    fetal_field_key = "field_fetal" if entity == "fetal" else "field_maternal"
                    if fetal_field_key in ica_beat_avgs:
                        i_avg = ica_beat_avgs[fetal_field_key]
                        entry["mean_beat_ica"] = i_avg["mean"][:, ch_mask]
                        entry["std_beat_ica"]  = i_avg["std"][:, ch_mask]

            entity_beats[key] = entry

        heartbeats_dict[entity] = entity_beats

    return {"data_dict": data_dict, "heartbeats_dict": heartbeats_dict}


def build_hierarchical_format(preprocessed, approach_results, unified_results, config):
    """
    Build the pipeline_results_v2.pkl hierarchical dict.

    Parameters
    ----------
    preprocessed : PreprocessedData
    approach_results : dict
        {approach_name: ApproachResult}
    unified_results : dict or None
        {approach_name: unified_beat_dict} from _average_with_peaks
    config : dict
    """
    approaches = {}
    for name, result in approach_results.items():
        if result is None:
            continue
        approaches[name] = {
            "outputs": {
                k: v for k, v in result.domains.items()
                if isinstance(v, np.ndarray)
            },
            "metadata": result.metadata,
            "postprocessing": result.metadata.get("postprocessing"),
        }

    return {
        "metadata": {
            "version": "2.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
        },
        "preprocessing": {
            "fs": preprocessed.fs,
            "artifacts_mask": preprocessed.artifacts_mask,
            "axis_mask": preprocessed.axis_mask,
            "whitening_matrix": preprocessed.whitening_matrix,
            "time": preprocessed.time,
        },
        "approaches": approaches,
        "unified": unified_results,
    }


def load_results(path):
    """
    Load a results file, auto-detecting legacy vs hierarchical format.

    Returns
    -------
    dict
        The loaded results dict.
    str
        "legacy" or "hierarchical"
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "metadata" in data and data.get("metadata", {}).get("version"):
        return data, "hierarchical"
    return data, "legacy"
