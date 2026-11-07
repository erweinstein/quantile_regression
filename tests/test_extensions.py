'''tests/test_extensions.py -- checks for qr_extensions.py.'''
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rq import rq  # noqa: E402
from selection_qr import probit, selection_qr  # noqa: E402
from qr_extensions import (rearrange, rq_se, selection_qr_weighted,  # noqa: E402
                           weighted_probit, weighted_rq)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1 + 2 * x + rng.standard_t(5, size=n)
    return np.column_stack([np.ones(n), x]), y, rng


def test_weighted_rq_equals_replicated_data():
    X, y, rng = _data()
    w = rng.integers(0, 4, size=len(y)).astype(float)
    idx = np.repeat(np.arange(len(y)), w.astype(int))
    assert np.allclose(weighted_rq(X, y, 0.3, w), rq(X[idx], y[idx], 0.3), atol=1e-6)


def test_weighted_probit_with_unit_weights_is_probit():
    X, y, rng = _data()
    D = (X[:, 1] + rng.normal(size=len(y)) > 0).astype(int)
    assert np.allclose(weighted_probit(D, X[:, 1:], np.ones(len(D))), probit(D, X[:, 1:]))


def test_rq_se_positive_and_shrinks_with_n():
    X, y, _ = _data(400)
    X2, y2, _ = _data(4000, seed=1)
    for m in ('ker', 'iid'):
        se_small, _ = rq_se(X, y, 0.5, method=m)
        se_large, _ = rq_se(X2, y2, 0.5, method=m)
        assert np.all(se_small > 0) and np.all(se_large < se_small)


def test_selection_qr_weighted_reproduces_selection_qr():
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=n)
    B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    Y = 1 + X + (0.5 + 0.3 * X) * u
    a = selection_qr(D, Y, X, B, vecrho=np.round(np.arange(-0.9, 0.91, 0.1), 2))
    b = selection_qr_weighted(D, Y, X, B, vecrho=np.round(np.arange(-0.9, 0.91, 0.1), 2))
    assert a['rho'] == b['rho'] and np.allclose(a['beta'], b['beta'])


def test_rearrange_is_monotone_and_a_permutation():
    beta = np.array([[0.0, 0.5, 0.4, 1.0], [1.0, 0.8, 1.1, 0.9]])
    x = np.array([1.0, 2.0])
    q = x @ beta
    r = rearrange(beta, x)
    assert np.all(np.diff(r) >= 0) and np.allclose(np.sort(q), r)
