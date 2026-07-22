"""Loader for the St Petersburg INCART 12-lead Arrhythmia Database."""

import os
import re
import warnings
from typing import Optional

import numpy as np
import wfdb

from ._common import BEAT_SYMBOLS as _BEAT_SYMBOLS

# I01 .. I75, 30-min 12-lead recordings from ischemic patients.
INCART_RECORDS = [f"I{i:02d}" for i in range(1, 76)]


def _parse_diagnosis(comments: list) -> str:
    """Pull the free-text diagnosis from the header ``<diagnoses>`` comment."""
    for c in comments:
        m = re.search(r"<diagnoses>\s*(.*)", c)
        if m:
            return m.group(1).strip()
    return ""


def load_incart(
    data_path: Optional[str] = None,
    records: Optional[list] = None,
    max_records: Optional[int] = None,
) -> list[dict]:
    """Load records from the St Petersburg INCART database.

    Parameters
    ----------
    data_path : str, optional
        Path to a local directory containing INCART ``.hea`` / signal /
        annotation files. When ``None`` the records are fetched from PhysioNet
        (``incartdb``, requires internet access).
    records : list, optional
        Subset of record names (e.g. ``["I01", "I02"]``). Loads all 75 when
        ``None``.
    max_records : int, optional
        Cap the number of records loaded.

    Returns
    -------
    list[dict]
        One dict per record with keys:

        - ``signal``      : ndarray (T, 12) float64, mV
        - ``r_peaks``     : ndarray of R-peak sample indices
        - ``beat_labels`` : list[str] — beat-type symbol at each R-peak
        - ``diagnosis``   : str — free text from the header ``<diagnoses>``
        - ``record_name`` : str
        - ``fs``          : int  (257 for INCART)
        - ``lead_names``  : list[str] — 12 leads (I, II, III, aVR ... V6)
    """
    record_list = list(records) if records is not None else list(INCART_RECORDS)
    if max_records is not None:
        record_list = record_list[:max_records]

    out = []
    for name in record_list:
        try:
            if data_path is None:
                rec = wfdb.rdrecord(name, pn_dir="incartdb")
                ann = wfdb.rdann(name, extension="atr", pn_dir="incartdb")
            else:
                path = os.path.join(data_path, name)
                rec = wfdb.rdrecord(path)
                ann = wfdb.rdann(path, extension="atr")

            signal = rec.p_signal  # (T, 12) float64, mV
            lead_names = list(rec.sig_name)

            beat_mask = np.array([s in _BEAT_SYMBOLS for s in ann.symbol])
            r_peaks = ann.sample[beat_mask].astype(int)
            beat_labels = [ann.symbol[i] for i in np.where(beat_mask)[0]]

            out.append({
                "signal": signal,
                "r_peaks": r_peaks,
                "beat_labels": beat_labels,
                "diagnosis": _parse_diagnosis(rec.comments),
                "record_name": name,
                "fs": rec.fs,
                "lead_names": lead_names,
            })
        except Exception as e:
            warnings.warn(f"Skipping INCART record '{name}': {e}")

    return out
