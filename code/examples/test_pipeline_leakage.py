"""Example: PAT as an ordinary unit test that runs in continuous integration (pytest).

The test executes the project's feature-pipeline script unchanged. It first checks that the
pipeline is deterministic, then audits the first and last rows of the test block. A failing
test names the input columns that depend on information unavailable at prediction time.

    pytest -q examples/test_pipeline_leakage.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
from pat_core import ScriptPipeline, pat_temporal, determinism_check   # noqa: E402

N_TEST = 56


def _series():
    rng = np.random.default_rng(0)
    t = np.arange(800)
    y = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.cumsum(rng.normal(size=t.size))
    return pd.DataFrame({'y': np.maximum(y, 0)})


def _audit(script):
    pipe = ScriptPipeline(os.path.join(HERE, script))
    df = _series()
    nondet = determinism_check(pipe, df)
    assert not nondet, f'pipeline is nondeterministic in {nondet}; fix its seeds first'
    res = pat_temporal(pipe, df, [len(df) - N_TEST, len(df) - 1], B=5, scale=2.0, seed=0)
    return sorted({c for a in res['audits'] for c in a['flagged']})


def test_clean_pipeline_is_admissible():
    assert _audit('pipeline_clean.py') == []


def test_leaky_pipeline_is_reported_with_its_columns():
    flagged = _audit('pipeline_leaky.py')
    assert flagged, 'PAT should flag the centred rolling mean'
    assert all('roll_mean' in c for c in flagged)
