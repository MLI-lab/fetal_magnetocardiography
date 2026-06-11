"""Loader for the ISP ECG delineation dataset."""

import ast
import os
import warnings
from typing import Literal, Optional

import pandas as pd
import wfdb

from .utils import annotations_to_mask


def load_isp(data_path: str, split: Literal["train", "test"], records: Optional[list] = None, max_records: Optional[int] = None) -> list[dict]:
    """Load records from the ISP ECG delineation dataset.

    Parameters
    ----------
    data_path : str
        Root directory containing ``train_data/``, ``test_data/`` subdirs
        and the corresponding CSV files.
    split : {'train', 'test'}
        Which split to load.
    records : list, optional
        Subset of record names to load. Loads all records in the split CSV
        when ``None``.
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
        - ``age``         : int or None
        - ``sex``         : int or None
    """
    data_dir = os.path.join(data_path, f"{split}_data")
    csv_path = os.path.join(data_path, f"{split}_isp_delineation_data.csv")

    df = pd.read_csv(csv_path)
    if records is not None:
        df = df[df["file_name"].isin([str(r) for r in records])]
    if max_records is not None:
        df = df.head(max_records)

    out = []
    for _, row in df.iterrows():
        record_name = str(row["file_name"])
        try:
            rec = wfdb.rdrecord(os.path.join(data_dir, record_name))
            signal = rec.p_signal  # (T, C) float64, mV
            lead_names = list(rec.sig_name)

            raw = ast.literal_eval(row["target"])
            annotations = [
                {"wave_type": wave_type, "start": start, "end": end, "channel": ch}
                for wave_type, start, end in raw
                for ch in range(signal.shape[1])
            ]
            labels = annotations_to_mask(signal.shape[0], signal.shape[1], annotations)

            out.append(
                {
                    "signal": signal,
                    "labels": labels,
                    "record_name": record_name,
                    "fs": rec.fs,
                    "lead_names": lead_names,
                    "age": int(row["age"]) if pd.notna(row["age"]) else None,
                    "sex": int(row["sex"]) if pd.notna(row["sex"]) else None,
                }
            )
        except Exception as e:
            warnings.warn(f"Skipping ISP record '{record_name}': {e}")

    return out
