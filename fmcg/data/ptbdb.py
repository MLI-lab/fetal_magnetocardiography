"""Loader for the PTB Diagnostic ECG Database (PTB-DB) VCG signals."""

import logging
import os

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


def load_vcg_data(
    record="patient104/s0306lre", base_path=None, plot=False
):
    """Load and preprocess VCG (Vectorcardiogram) data from a specified record.

    Parameters
    ----------
    record : str
        Path to the record file within the dataset.
        Default is ``"patient104/s0306lre"``.
    base_path : str, optional
        Base path to a local PTB-DB dataset directory. When ``None`` the
        record is fetched from the PhysioNet-hosted PTB-DB.
    plot : bool
        If ``True``, plots the VCG signals. Default is ``False``.

    Returns
    -------
    filtered_signals : ndarray
        Bandpass-filtered VCG signals (vx, vy, vz).
    ecg_II : ndarray
        ECG lead II signal.
    fs : int
        Sampling frequency of the record.
    """
    if base_path is None:
        record_dir = os.path.dirname(record)
        record_base = os.path.basename(record)
        pn_dir = f"ptbdb/1.0.0/{record_dir}" if record_dir else "ptbdb/1.0.0"
        try:
            rec = wfdb.rdrecord(record_base, physical=True, pn_dir=pn_dir)
        except Exception as e:
            logger.warning(
                "Failed to load record '%s' from PhysioNet (pn_dir=%s): %s",
                record_base, pn_dir, e,
            )
            raise ValueError(
                f"Could not load record '{record}' from PhysioNet. "
                "Provide a local base_path or check the record name. "
                f"Error: {e}"
            ) from e
    else:
        rec = wfdb.rdrecord(f"{base_path}/{record}", physical=True)

    fs = rec.fs

    indices = np.argsort(np.argsort(["vx", "vy", "vz"]))
    sorted_indices = np.where(np.in1d(rec.sig_name, ["vx", "vy", "vz"]))[0][indices]
    signals = rec.p_signal[:, sorted_indices]
    ecg_II = rec.p_signal[:, 1]

    lowcut, highcut = 0.5, 40.0
    nyquist = 0.5 * fs
    b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype="band")
    filtered_signals = filtfilt(b, a, signals, axis=0)

    if plot:
        from fmcg.utils.plotting import plot_utils
        plot_utils.plot_vcg(filtered_signals, ecg_II, fs)

    return filtered_signals, ecg_II, fs
