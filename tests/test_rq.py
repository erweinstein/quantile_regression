'''tests/test_rq.py -- checks for rq.py and selection_qr.py.

Run from the project folder with:  python -m pytest tests
'''

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rq import rq, rqic, rq_objective, bound            # noqa: E402
from selection_qr import bivariate_normal_cdf, probit, selection_qr  # noqa: E402


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(size=n)])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.standard_t(3, size=n) * (1 + X[:, 2])
    return X, y


def lp_solution(X, y, tau):
    '''The primal LP solved by a different algorithm (HiGHS, via scipy):
    min tau 1'u + (1 - tau) 1'v  s.t.  X b + u - v = y,  u, v >= 0.'''
    n, p = X.shape
    c = np.r_[np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)]
    A = np.hstack([X, np.eye(n), -np.eye(n)])
    res = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * p + [(0, None)] * (2 * n),
                  method='highs')
    return res.x[:p], res.fun


def test_bound():
    x = np.array([1.0, 2.0, 3.0])
    dx = np.array([-0.5, 1.0, -3.0])
    assert np.allclose(bound(x, dx), [2.0, 1e20, 1.0])


@pytest.mark.parametrize('tau', [0.1, 0.25, 0.5, 0.75, 0.9])
def test_rq_matches_linear_programming(tau):
    X, y = data()
    b = rq(X, y, tau)
    _, fun = lp_solution(X, y, tau)
    # Solutions need not be unique (Quantreg FAQ 1), but the objective is.
    assert abs(rq_objective(X, y, b, tau) - fun) < 1e-6 * max(1, fun)


def test_rq_median_of_constant():
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    b = rq(np.ones((7, 1)), y, 0.5)
    assert abs(b[0] - 3.0) < 1e-4                          # the sample median


def test_rqic_slack_and_binding():
    X, y = data()
    b0 = rq(X, y, 0.5)
    slack = rqic(X, y, [[0, 0, 1]], [-10.0], 0.5)       # constraint not binding
    assert np.allclose(slack, b0, atol=1e-4)
    bind = rqic(X, y, [[0, 0, 1]], [0.5], 0.5)          # forces b3 >= 0.5
    assert bind[2] >= 0.5 - 1e-6


def test_rq_input_checks():
    X, y = data()
    with pytest.raises(ValueError):
        rq(X, y, 1.5)
    with pytest.raises(ValueError):
        rq(np.column_stack([X, X[:, 1]]), y, 0.5)       # rank-deficient design


@pytest.mark.parametrize('rho', [-0.9, -0.3, 0.0, 0.5, 0.95])
def test_bivariate_normal_cdf(rho):
    rng = np.random.default_rng(1)
    h, k = rng.normal(size=15), rng.normal(size=15)
    h[0], k[1] = 0.0, 0.0
    ref = [multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, rho], [rho, 1]])
           for a, b in zip(h, k)]
    assert np.max(np.abs(bivariate_normal_cdf(h, k, rho) - ref)) < 1e-7


def test_selection_qr_recovers_simulated_model():
    # Y* = 1 + X + eps, D = 1{0.5 + 0.5 X + B + nu > 0}, corr(eps, nu) = 0.5.
    # Arellano and Bonhomme's copula links the outcome rank to the rank V of
    # the selection error with D = 1{V <= p(Z)}, i.e. V = Phi(-nu), so the
    # copula parameter is -0.5 here.
    rng = np.random.default_rng(1)
    n, rho0 = 2000, 0.5
    X, B = rng.normal(size=n), rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1, rho0], [rho0, 1]], size=n)
    D = (0.5 + 0.5 * X + B + e[:, 1] > 0).astype(int)
    Y = 1 + X + e[:, 0]
    g = probit(D, np.column_stack([X, B]))
    assert np.allclose(g, [0.5, 0.5, 1.0], atol=0.15)
    r = selection_qr(D, Y, X, B)
    assert abs(r['rho'] - (-rho0)) < 0.15
    j = list(np.round(r['taus'], 2)).index(0.5)
    assert abs(r['beta'][0, j] - 1.0) < 0.15 and abs(r['beta'][1, j] - 1.0) < 0.15
