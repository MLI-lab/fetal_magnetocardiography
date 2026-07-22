"""Loader for the MIT-BIH Normal Sinus Rhythm Database (nsrdb)."""

import os
import warnings
from typing import Optional

import numpy as np
import wfdb

from ._common import BEAT_SYMBOLS as _BEAT_SYMBOLS

# 18 long-term (~24 h) Holter recordings from subjects with no arrhythmia.
NSRDB_RECORDS = [
    "16265", "16272", "16273", "16420", "16483", "16539",
    "16773", "16786", "16795", "17052", "17453", "18177",
    "18184", "19088", "19090", "19093", "19140", "19830",
]


def load_nsrdb(
    data_path: Optional[str] = None,
    records: Optional[list] = None,
    max_records: Optional[int] = None,
    win_min: Optional[float] = 30.0,
    win_start_min: float = 0.0,
) -> list[dict]:
    """Load records from the MIT-BIH Normal Sinus Rhythm Database.

    The recordings are ~24 h Holter traces. To length-match the shorter
    databases (INCART, MIT-BIH), a single fixed window of ``win_min`` minutes
    starting at ``win_start_min`` is extracted; pass ``win_min=None`` for the
    full recording.

    Parameters
    ----------
    data_path : str, optional
        Path to a local directory containing nsrdb ``.hea`` / signal /
        annotation files. When ``None`` the records are fetched from PhysioNet
        (``nsrdb``, requires internet access).
    records : list, optional
        Subset of record names. Loads all 18 when ``None``.
    max_records : int, optional
        Cap the number of records loaded.
    win_min : float, optional
        Length of the extracted window in minutes (default 30). ``None`` keeps
        the full recording.
    win_start_min : float
        Start of the window in minutes from the recording start (default 0).

    Returns
    -------
    list[dict]
        One dict per record with keys:

        - ``signal``      : ndarray (T, 2) float64, mV
        - ``r_peaks``     : ndarray of R-peak sample indices (relative to window)
        - ``beat_labels`` : list[str] — beat-type symbol at each R-peak
        - ``record_name`` : str
        - ``fs``          : int  (128 for nsrdb)
        - ``lead_names``  : list[str]  (``["ECG1", "ECG2"]``)
    """
    record_list = list(records) if records is not None else list(NSRDB_RECORDS)
    if max_records is not None:
        record_list = record_list[:max_records]

    out = []
    for name in record_list:
        try:
            if data_path is None:
                rec = wfdb.rdrecord(name, pn_dir="nsrdb")
                ann = wfdb.rdann(name, extension="atr", pn_dir="nsrdb")
            else:
                path = os.path.join(data_path, name)
                rec = wfdb.rdrecord(path)
                ann = wfdb.rdann(path, extension="atr")

            signal = rec.p_signal  # (T, 2) float64, mV
            lead_names = list(rec.sig_name)
            fs = rec.fs

            beat_mask = np.array([s in _BEAT_SYMBOLS for s in ann.symbol])
            r_peaks = ann.sample[beat_mask].astype(int)
            beat_labels = [ann.symbol[i] for i in np.where(beat_mask)[0]]

            if win_min is not None:
                lo = int(win_start_min * 60 * fs)
                hi = lo + int(win_min * 60 * fs)
                signal = signal[lo:hi]
                keep = (r_peaks >= lo) & (r_peaks < hi)
                r_peaks = r_peaks[keep] - lo
                beat_labels = [b for b, k in zip(beat_labels, keep) if k]

            out.append({
                "signal": signal,
                "r_peaks": r_peaks,
                "beat_labels": beat_labels,
                "record_name": name,
                "fs": fs,
                "lead_names": lead_names,
            })
        except Exception as e:
            warnings.warn(f"Skipping nsrdb record '{name}': {e}")

    return out
