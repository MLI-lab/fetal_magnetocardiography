"""Loader for the Lobachevsky University ECG Database (LUDB)."""

import glob
import os
import warnings
from typing import Optional

import numpy as np
import wfdb

from .utils import annotations_to_mask

# LUDB annotation symbols
_SYMBOL_TO_LABEL = {"p": 1, "N": 2, "t": 3}  # P=1, QRS=2, T=3; 0=background


def _extract_r_peaks(data_path: str, record_name: str, lead_names: list) -> np.ndarray:
    """Return R-peak sample indices from LUDB annotations (the 'N' symbol).

    Collects 'N' samples across all leads and collapses per-lead jitter
    (< 5 samples) to a single median position per beat.
    """
    all_samples = []
    for lead in lead_names:
        try:
            ann = wfdb.rdann(os.path.join(data_path, record_name), extension=lead)
        except Exception:
            continue
        all_samples.extend(s for sym, s in zip(ann.symbol, ann.sample) if sym == "N")

    if not all_samples:
        return np.array([], dtype=int)

    samples = np.sort(all_samples)
    gaps = np.diff(samples)
    breaks = np.where(gaps > 15)[0] + 1
    clusters = np.split(samples, breaks)
    return np.array([int(np.median(c)) for c in clusters], dtype=int)


def load_ludb(data_path: str, records: Optional[list] = None, max_records: Optional[int] = None) -> list[dict]:
    """Load all records from the LUDB dataset.

    Parameters
    ----------
    data_path : str
        Path to the directory containing LUDB `.hea` / signal / annotation files.
    records : list, optional
        Subset of record names (without extension) to load. Loads all records
        found in ``data_path`` when ``None``.
    max_records : int, optional
        Cap the number of records loaded.

    Returns
    -------
    list[dict]
        One dict per record with keys:
        - ``signal``      : ndarray (T, C) float64, mV
        - ``labels``      : ndarray (T, C) int64 — 0=bg, 1=P, 2=QRS, 3=T
        - ``record_name`` : str
        - ``fs``          : int
        - ``lead_names``  : list[str]
    """
    if records is None:
        hea_files = sorted(glob.glob(os.path.join(data_path, "*.hea")))
        record_list = [os.path.basename(f).replace(".hea", "") for f in hea_files]
    else:
        record_list = [str(r) for r in records]

    if max_records is not None:
        record_list = record_list[:max_records]

    out = []
    for name in record_list:
        try:
            rec = wfdb.rdrecord(os.path.join(data_path, name))
            signal = rec.p_signal  # (T, C) float64, mV
            lead_names = list(rec.sig_name)

            annotations = _parse_annotations(data_path, name, lead_names)
            labels = annotations_to_mask(signal.shape[0], signal.shape[1], annotations)

            r_peaks = _extract_r_peaks(data_path, name, lead_names)

            out.append(
                {
                    "signal": signal,
                    "labels": labels,
                    "record_name": name,
                    "fs": rec.fs,
                    "lead_names": lead_names,
                    "r_peaks": r_peaks,
                }
            )
        except Exception as e:
            warnings.warn(f"Skipping LUDB record '{name}': {e}")

    return out


def _parse_annotations(data_path: str, record_name: str, lead_names: list) -> list[dict]:
    """Parse LUDB wave-boundary annotations for all leads.

    Returns a flat list of dicts with keys ``wave_type``, ``start``, ``end``,
    ``channel``.
    """
    annotations = []
    for ch_idx, lead in enumerate(lead_names):
        try:
            ann = wfdb.rdann(os.path.join(data_path, record_name), extension=lead)
        except Exception:
            continue  # annotation file absent for this lead — skip

        onset = label = None
        for sym, sample in zip(ann.symbol, ann.sample):
            if sym == "(":
                onset, label = sample, None
            elif sym in _SYMBOL_TO_LABEL:
                label = _SYMBOL_TO_LABEL[sym]
            elif sym == ")" and onset is not None and label is not None:
                annotations.append(
                    {"wave_type": label, "start": int(onset), "end": int(sample), "channel": ch_idx}
                )
                onset = label = None

    return annotations


