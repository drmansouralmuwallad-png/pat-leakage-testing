"""Series data for the temporal corpus: the 150 series of the four public panels.

Pipelines with an imputation stage (C1, C5) receive a reporting-gap mask. The mask covers 2%
of positions at random, plus the position immediately before each audited row, which models
a value that has not yet been reported. The mask is fixed by the series id, so it is part of
the data, not of the perturbation.
"""
import zlib

import numpy as np
import pandas as pd

from corpus_temporal import PANELS

IMPUTING_CONFIGS = {'C1_basic', 'C5_chain'}


def load_panels(path):
    return np.load(path, allow_pickle=True).item()


def audit_rows(n, n_test):
    """First and last row of the test block."""
    return [n - n_test, n - 1]


def series_frame(panel, sid, y, masked):
    y = np.asarray(y, dtype=float).copy()
    if masked:
        n = len(y)
        c = PANELS[panel]
        rng = np.random.default_rng(zlib.crc32(f'{panel}|{sid}'.encode()))
        warm = 3 * c['M'] + 1
        audits = audit_rows(n, c['N_TEST'])
        cand = np.setdiff1d(np.arange(warm, n), audits)
        pos = rng.choice(cand, size=int(0.02 * n), replace=False)
        y[pos] = np.nan
        for a in audits:
            y[a - 1] = np.nan
    return pd.DataFrame({'y': y})
