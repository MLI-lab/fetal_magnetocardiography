"""Shared ICA annotation file discovery and loading utilities."""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ICA_FILENAME_PATTERN = re.compile(
    r"_P(?P<patient>\d+)_S(?P<series>\d+)_(?P<signal_group>[A-Za-z]\d+)_"
)


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _normalize_patient(patient: str) -> str:
    text = str(patient).strip()
    if text.upper().startswith("P"):
        text = text[1:]
    return text


def _normalize_series(series: str) -> str:
    text = str(series).strip()
    if text.upper().startswith("S"):
        text = text[1:]
    return text


def _normalize_signal_group(signal_group: str) -> str:
    text = str(signal_group).strip().upper()
    if not text:
        return text
    if text[0].isalpha():
        return text
    if text.isdigit():
        return f"R{text.zfill(3)}"
    return text


def _extract_ids_from_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    match = _ICA_FILENAME_PATTERN.search(filename)
    if match is not None:
        return (
            _normalize_patient(match.group("patient")),
            _normalize_series(match.group("series")),
            _normalize_signal_group(match.group("signal_group")),
        )

    patient = None
    series = None
    signal_group = None
    for part in filename.replace(".npz", "").split("_"):
        if part.startswith("P") and part[1:].isdigit():
            patient = _normalize_patient(part)
        elif part.startswith("S") and part[1:].isdigit():
            series = _normalize_series(part)
        elif len(part) >= 2 and part[0].isalpha() and part[1:].isdigit():
            signal_group = _normalize_signal_group(part)

    if patient and series and signal_group:
        return patient, series, signal_group
    return None


def load_ica_npz(path: str) -> Dict[str, Any]:
    """Load NPZ/object payload and unwrap object scalars."""
    loaded = np.load(path, allow_pickle=True)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        out: Dict[str, Any] = {}
        for key in loaded.files:
            out[key] = _unwrap_scalar(loaded[key])
        loaded.close()
        return out

    if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
        value = loaded.item()
        if isinstance(value, dict):
            return value

    raise ValueError(f"Unsupported ICA payload format: {path}")


def load_exported_ica_data(npz_file: str, strict: bool = False) -> Dict[str, Any]:
    """Load ICAComponentLabeler export format with optional strict key checks."""
    payload = load_ica_npz(npz_file)

    labels_dict = payload.get("labels_dict")
    if labels_dict is None and payload.get("component_labels") is not None:
        component_labels = np.asarray(payload["component_labels"]).astype(str)
        labels_dict = {i: label for i, label in enumerate(component_labels)}

    result: Dict[str, Any] = {
        "sources": payload.get("sources"),
        "mixing_matrix": payload.get("mixing_matrix"),
        "labels_dict": labels_dict,
        "fs": payload.get("fs"),
        "component_labels": payload.get("component_labels"),
        "component_order": payload.get("component_order"),
        "bp_data": payload.get("bp_data"),
        "axis_mask": payload.get("axis_mask"),
        "ica": payload.get("ica"),
    }

    for optional_key in ("fetal_reference", "maternal_reference", "all_peaks", "seed"):
        if optional_key in payload:
            result[optional_key] = payload[optional_key]

    if strict:
        required_keys = (
            "sources",
            "mixing_matrix",
            "labels_dict",
            "fs",
            "component_labels",
            "bp_data",
            "axis_mask",
            "ica",
        )
        missing = [key for key in required_keys if result.get(key) is None]
        if missing:
            raise KeyError(
                f"Missing required ICA export keys in {npz_file}: {', '.join(missing)}"
            )

    return result


def build_ica_file_cache(ica_annotation_path: str) -> Dict[Tuple[str, str, str], str]:
    """Build lookup cache: (patient, series, signal_group) -> annotation file path."""
    cache: Dict[Tuple[str, str, str], str] = {}
    if not os.path.exists(ica_annotation_path):
        return cache

    for filepath in sorted(glob.glob(os.path.join(ica_annotation_path, "ica_data_*.npz"))):
        ids = _extract_ids_from_filename(os.path.basename(filepath))
        if ids is None:
            continue
        cache[ids] = filepath

    return cache


def find_ica_annotation_file(
    patient: str,
    series: str,
    signal_group: str,
    ica_annotation_path: str = "/data/ica_annotation/ica_annotation_jonas10_full",
    ica_file_cache: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> Optional[str]:
    """Find ICA annotation file for patient/series/signal-group triple."""
    patient_id = _normalize_patient(patient)
    series_id = _normalize_series(series)
    signal_group_id = _normalize_signal_group(signal_group)
    key = (patient_id, series_id, signal_group_id)

    if ica_file_cache is not None:
        return ica_file_cache.get(key)

    if not os.path.exists(ica_annotation_path):
        return None

    prefixed_patient = f"P{patient_id}"
    prefixed_series = f"S{series_id}"
    pattern_extra = os.path.join(
        ica_annotation_path,
        f"ica_data_*_{prefixed_patient}_{prefixed_series}_{signal_group_id}_*.npz",
    )
    pattern_exact = os.path.join(
        ica_annotation_path,
        f"ica_data_*_{prefixed_patient}_{prefixed_series}_{signal_group_id}.npz",
    )

    matches = sorted(glob.glob(pattern_extra) + glob.glob(pattern_exact))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple ICA annotation matches for %s_%s_%s; using first match.",
            prefixed_patient,
            prefixed_series,
            signal_group_id,
        )
    return matches[0]


def get_ica_records_from_path(path: str) -> pd.DataFrame:
    """Return all ICA annotation record triples as a dataframe."""
    cache = build_ica_file_cache(path)
    records = sorted(cache.keys())
    return pd.DataFrame(records, columns=["patient", "series", "signal_group"])


__all__ = [
    "build_ica_file_cache",
    "find_ica_annotation_file",
    "get_ica_records_from_path",
    "load_exported_ica_data",
    "load_ica_npz",
    "load_ica_npz",
]
