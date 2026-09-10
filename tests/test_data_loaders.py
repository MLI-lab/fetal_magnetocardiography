"""Label contract of the WFDB delineation loaders.

Both loaders return a dense segmentation mask whose background is 0, so the
wave classes have to be one-based. ISP's CSV is zero-based and needs a shift,
LUDB's symbol map already emits the right values. Neither had a test before,
which is how the ISP shift went missing.
"""

import os

import numpy as np
import pytest

from fmcg.data import load_isp, load_ludb

ISP_PATH = "/ECG/isp_delineation"
LUDB_PATH = "/ECG/ludb/1.0.1/data"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(ISP_PATH) and os.path.isdir(LUDB_PATH)),
    reason="needs the ECG databases on disk",
)


def _label_values(records):
    return set(np.unique(np.concatenate([r["labels"].ravel() for r in records])).tolist())


def test_isp_labels_follow_the_documented_contract():
    records = load_isp(ISP_PATH, split="train", max_records=5)
    assert _label_values(records) == {0, 1, 2, 3}


def test_ludb_labels_follow_the_documented_contract():
    records = load_ludb(LUDB_PATH, max_records=5)
    assert _label_values(records) == {0, 1, 2, 3}


def test_isp_wave_order_within_a_beat_is_p_qrs_t():
    """Within one beat the classes occur as 1, then 2, then 3.

    Checked from the first P wave onward rather than from the start of the
    record, because a recording can begin part way through a beat. A shift in
    the wrong direction, or none at all, breaks this.
    """
    labels = load_isp(ISP_PATH, split="train", max_records=1)[0]["labels"][:, 0]
    p_start = int(np.flatnonzero(labels == 1)[0])
    after = labels[p_start:]
    qrs = int(np.flatnonzero(after == 2)[0])
    t_wave = int(np.flatnonzero(after == 3)[0])
    p_end = int(np.flatnonzero(after != 1)[0])
    assert p_end < qrs < t_wave
