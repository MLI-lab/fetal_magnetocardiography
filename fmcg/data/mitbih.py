"""Loader for the MIT-BIH Arrhythmia Database."""

import os
import warnings
from typing import Optional

import numpy as np
import wfdb

from ._common import BEAT_SYMBOLS as _BEAT_SYMBOLS

MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
]


def load_mitbih(
    data_path: Optional[str] = None,
    records: Optional[list] = None,
    max_records: Optional[int] = None,
) -> list[dict]:
    """Load records from the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    data_path : str, optional
        Path to a local directory containing MIT-BIH ``.hea`` / signal /
        annotation files. When ``None`` the records are fetched directly from
        PhysioNet (requires internet access).
    records : list, optional
        Subset of record names (e.g. ``["100", "101"]``) to load.  Loads all
        48 standard records when ``None``.
    max_records : int, optional
        Cap the number of records loaded.

    Returns
    -------
    list[dict]
        One dict per record with keys:

        - ``signal``      : ndarray (T, 2) float64, mV
        - ``r_peaks``     : ndarray of R-peak sample indices
        - ``beat_labels`` : list[str] — beat-type symbol at each R-peak
        - ``rhythms``     : list[str] — unique rhythm codes seen in this record
        - ``record_name`` : str
        - ``fs``          : int  (360 for MIT-BIH)
        - ``lead_names``  : list[str]
    """
    record_list = list(records) if records is not None else list(MITBIH_RECORDS)
    if max_records is not None:
        record_list = record_list[:max_records]

    out = []
    for name in record_list:
        try:
            if data_path is None:
                rec = wfdb.rdrecord(name, pn_dir="mitdb")
                ann = wfdb.rdann(name, extension="atr", pn_dir="mitdb")
            else:
                path = os.path.join(data_path, name)
                rec = wfdb.rdrecord(path)
                ann = wfdb.rdann(path, extension="atr")

            signal = rec.p_signal  # (T, 2) float64, mV
            lead_names = list(rec.sig_name)

            # Beat annotations: filter to actual heartbeat symbols
            beat_mask = np.array([s in _BEAT_SYMBOLS for s in ann.symbol])
            r_peaks = ann.sample[beat_mask].astype(int)
            beat_labels = [ann.symbol[i] for i in np.where(beat_mask)[0]]

            # Rhythm codes: aux_note entries starting with '(' mark rhythm changes;
            # strip '(' prefix and null bytes that wfdb sometimes appends.
            rhythms = list(dict.fromkeys(
                note.lstrip("(").rstrip("\x00").strip()
                for note, sym in zip(ann.aux_note, ann.symbol)
                if note and note.startswith("(")
            ))

            out.append({
                "signal": signal,
                "r_peaks": r_peaks,
                "beat_labels": beat_labels,
                "rhythms": rhythms,
                "record_name": name,
                "fs": rec.fs,
                "lead_names": lead_names,
            })
        except Exception as e:
            warnings.warn(f"Skipping MIT-BIH record '{name}': {e}")

    return out
