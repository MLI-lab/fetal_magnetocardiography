"""Shared helpers for ECG data loaders."""

import numpy as np


def annotations_to_mask(signal_length: int, num_channels: int, annotations: list) -> np.ndarray:
    """Convert a flat annotation list to an integer label mask.

    Parameters
    ----------
    annotations : list[dict]
        Each dict must have keys ``wave_type`` (int), ``start`` (int),
        ``end`` (int), ``channel`` (int).

    Returns
    -------
    np.ndarray
        Shape (signal_length, num_channels), dtype int64.
        Values: 0=background, 1=P, 2=QRS, 3=T.
    """
    mask = np.zeros((signal_length, num_channels), dtype=np.int64)
    for ann in annotations:
        ch = ann["channel"]
        start = max(0, ann["start"])
        end = min(signal_length, ann["end"])
        if start < end and ch < num_channels:
            mask[start:end, ch] = ann["wave_type"]
    return mask
