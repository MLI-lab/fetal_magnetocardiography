"""Loader for the PTB Diagnostic ECG Database (PTB-DB) VCG signals."""

import logging
import glob
import os
import warnings
from typing import Optional

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


def load_ptbdb(data_path: str, records: Optional[list] = None, max_records: Optional[int] = None) -> list[dict]:
    """Load records from the PTB Diagnostic ECG Database (PTB-DB).

    Parameters
    ----------
    data_path : str
        Path to the PTB-DB dataset directory.
    records : list of str, optional
        List of record names to load. If None, all records are loaded.
    max_records : int, optional
        Maximum number of records to load.

    Returns
    -------
    list of dict
        List of dictionaries containing the loaded record data.
    """
    if records is None:
        hea_files = sorted(glob.glob(os.path.join(data_path, "*/*.hea")))
        # Extract relative paths from subdirectories instead of full paths
        record_list = [f.replace(".hea", "").replace(data_path, "").lstrip(os.sep) for f in hea_files]
    else:
        record_list = [str(r) for r in records]
    if max_records is not None:
        record_list = record_list[:max_records]
    out = []
    for name in record_list:
        try:
            # Construct proper path: data_path + relative_path_from_subdirs
            rec_path = os.path.join(data_path, name)
            rec = wfdb.rdrecord(rec_path)
            out.append(
                {
                    "record_name": name,
                    "signal": rec.p_signal,
                    "lead_names": rec.sig_name,
                    "fs": rec.fs,
                    "comments": rec.comments,
                }
            )
        except Exception as e:
            warnings.warn(f"Skipping PTB-DB record '{name}': {e}")
    return out

def parse_diagnostic_class(comments):
    REASON_MAP = [
        ('myocardial infarction',   'Myocardial infarction'),
        ('heart failure',           'Cardiomyopathy/Heart failure'),
        ('cardiomyopathy',          'Cardiomyopathy/Heart failure'),
        ('bundle branch block',     'Bundle branch block'),
        ('dysrhythmia',             'Dysrhythmia'),
        ('arrhythmia',              'Dysrhythmia'),
        ('hypertrophy',             'Myocardial hypertrophy'),
        ('valvular heart disease',  'Valvular heart disease'),
        ('myocarditis',             'Myocarditis'),
        ('healthy control',         'Healthy controls'),
        ('healthy volunteer',       'Healthy controls'),
        ('palpitation',             'Miscellaneous'),
        ('stable angina',           'Miscellaneous'),
        ('unstable angina',         'Miscellaneous'),
    ]

    reason = None
    diagnose = None
    for line in comments:
        lower = line.lower()
        if lower.startswith('reason for admission:'):
            reason = lower.split(':', 1)[1].strip()
        elif lower.startswith('diagnose:'):
            diagnose = lower.split(':', 1)[1].strip()

    for text in [reason, diagnose]:
        if text and text != 'n/a':
            for key, cls in REASON_MAP:
                if key in text:
                    return cls

    return None  # unresolved — caller will propagate from patient

if __name__ == "__main__":
    all_ptb = load_ptbdb("/ECG/ptbdb")

    rows = []
    for rec in tqdm(all_ptb, total=len(all_ptb)):
        patient_id = rec['record_name'].split('/')[0]  # e.g. "patient045"
        rows.append({
            'record_name': rec['record_name'],
            'patient_id': patient_id,
            'diagnostic_class': parse_diagnostic_class(rec['comments']),
        })

    df_ptb = pd.DataFrame(rows)

    # propagate class within each patient (all recordings of a patient share one class)
    patient_class = (
        df_ptb.dropna(subset=['diagnostic_class'])
        .groupby('patient_id')['diagnostic_class']
        .first()
    )
    df_ptb['diagnostic_class'] = df_ptb['diagnostic_class'].fillna(
        df_ptb['patient_id'].map(patient_class)
    )

    print(df_ptb['diagnostic_class'].value_counts().to_string())
    print(f"\nStill unresolved: {df_ptb['diagnostic_class'].isna().sum()}")
    print(df_ptb[df_ptb['diagnostic_class'].isna()][['record_name', 'patient_id']])