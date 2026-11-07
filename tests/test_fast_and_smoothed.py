'''tests/test_fast_and_smoothed.py -- checks for selection_qr_fast.py and smoothed_qr.py.'''
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rq import rq  # noqa: E402
from selection_qr import selection_qr  # noqa: E402
from selection_qr_fast import (check_objective, rq_fast, rq_preprocessed,  # noqa: E402
                               selection_qr_fast)
from smoothed_qr import conquer  # noqa: E402

warnings.simplefilter('ignore')


def _problem(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    X = np.column_stack([np.ones(n), x])
    y = 1 + x @ [1, -0.5] + (1 + 0.5 * np.abs(x[:, 0])) * rng.standard_t(4, size=n)
    tau = rng.uniform(0.2, 0.8, size=n)
    return X, y, tau, rng


def test_rq_fast_matches_rq():
    X, y, tau, _ = _problem(800)
    assert np.allclose(rq_fast(X, y, tau), rq(X, y, tau), atol=1e-8)
    assert np.allclose(rq_fast(X, y, 0.3), rq(X, y, 0.3), atol=1e-8)


def test_preprocessing_reaches_the_full_optimum():
    X, y, tau, rng = _problem(6000, seed=1)
    b = rq(X, y, tau)
    bp, info = rq_preprocessed(X, y, tau, b + rng.normal(scale=0.05, size=3))
    assert not info['fallback'] and info['kept'] < 6000
    assert abs(check_objective(X, y, tau, bp) - check_objective(X, y, tau, b)) < 1e-6


def test_selection_qr_fast_full_grid_matches_original():
    rng = np.random.default_rng(3)
    n = 800
    X = rng.normal(size=n)
    B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    Y = 1 + X + (0.5 + 0.3 * X) * u
    grid = np.round(np.arange(-0.9, 0.91, 0.1), 2)
    a = selection_qr(D, Y, X, B, vecrho=grid)
    f = selection_qr_fast(D, Y, X, B, vecrho=grid)
    assert a['rho'] == f['rho'] and np.allclose(a['beta'], f['beta'], atol=1e-4)


def test_paper_settings():
    rng = np.random.default_rng(4)
    n = 500
    X = rng.normal(size=n)
    B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    Y = 1 + X + (0.5 + 0.3 * X) * u
    r = selection_qr_fast(D, Y, X, B, settings='paper', search='coarse')
    assert len(r['vecrho']) == 200 and -1 < r['rho'] < 1


def test_conquer_close_to_rq():
    X, y, _, _ = _problem(5000, seed=2)
    for tau in (0.25, 0.5, 0.75):
        assert np.max(np.abs(conquer(X, y, tau) - rq(X, y, tau))) < 0.05
