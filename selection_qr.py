'''selection_qr.py -- Selection-corrected quantile regression
(Arellano and Bonhomme's copula-based "rotated" quantile regression).

A Python port of Code_QRS.m, "Code for selection-corrected quantile
regression by Manuel Arellano and Stephane Bonhomme", which uses rq.m. The
method is from

    Arellano, M. and S. Bonhomme (2017), "Quantile Selection Models With an
    Application to Understanding Changes in Wage Inequality," Econometrica
    85(1), 1-28,

and is surveyed in Arellano and Bonhomme, "Sample Selection in Quantile
Regression: A Survey." Their empirical work uses, among others, the data of
Huber and Melly (2015, Journal of Applied Econometrics 30(7)).

Originally started for MSBI 31600 (Summer 2020) by Eliot R. Weinstein;
completed in 2026 (see README.md).

Copyright (c) 2020, 2026 Eliot R. Weinstein. Licensed under the BSD 3-Clause
License; see LICENSE and NOTICE.md (the license covers this port only, not the
original algorithms or MATLAB code of Koenker, Morillo, Eilers, Arellano, and
Bonhomme).

Inputs (as in Code_QRS.m)
-------------------------
    D : participation indicator (binary)
    Y : outcome (values of Y for D = 0 are arbitrary; they may even be missing)
    X : covariates, not including the constant
    B : excluded covariates, not including the constant;  Z = (X, B)

Specification in this version (the authors' defaults):
    propensity score:   probit
    copula:             Gaussian
    instrument function: varphi(Z) = propensity score p(Z)
    grid of tau's for estimating rho: tau = .2, .4, .6, .8
    estimates of beta(tau) for tau = .05, ..., .95
    grid search for rho with 99 equidistant values (-.98, ..., .98)
'''

import numpy as np
from scipy.special import owens_t
from scipy.stats import norm

from rq import rq


# =============================================================================
# Building blocks
# =============================================================================

def probit(D, Z, tol=1e-10, max_iter=100):
    '''Probit maximum likelihood by Fisher scoring (Newton's method with the
    expected information), with a constant added.

    MATLAB: gamma = glmfit(Z, D, 'binomial', 'link', 'probit');

    In 2020 I spent a long time choosing between scipy, statsmodels, and
    scikit-learn for this one line. The probit log-likelihood is globally
    concave, so a few Newton steps from zero are all it takes; writing it out
    keeps the dependencies to numpy and scipy and makes every step visible.
    Returns gamma = (constant, slopes), like glmfit.
    '''
    D = np.asarray(D, dtype=float)
    W = np.column_stack([np.ones(len(D)), np.asarray(Z, dtype=float)])
    g = np.zeros(W.shape[1])
    for _ in range(max_iter):
        xb = W @ g
        P = np.clip(norm.cdf(xb), 1e-12, 1 - 1e-12)
        phi = norm.pdf(xb)
        # score and (minus) Hessian of the probit log-likelihood
        lam = phi * (D - P) / (P * (1 - P))
        score = W.T @ lam
        h = phi ** 2 / (P * (1 - P))              # expected information weights (Fisher scoring)
        H = (W * h[:, None]).T @ W
        step = np.linalg.solve(H, score)
        g = g + step
        if np.max(np.abs(step)) < tol:
            break
    return g


def bivariate_normal_cdf(h, k, rho):
    '''Phi_2(h, k; rho), the standard bivariate normal cdf, vectorized, via
    Owen's T function:

        Phi_2(h, k; rho) = [Phi(h) + Phi(k)]/2 - T(h, a_h) - T(k, a_k) - c,
        a_h = (k - rho h) / (h sqrt(1 - rho^2)),  a_k = (h - rho k) / (k sqrt(1 - rho^2)),
        c = 0 if h k > 0, or h k = 0 with h + k >= 0;  c = 1/2 otherwise.

    This replaces MATLAB's copulacdf('Gaussian', ...), which has no direct
    scipy equivalent (scipy.stats.multivariate_normal.cdf works but is much
    slower, point by point). h = 0 or k = 0 (tau = 0.5 or p = 0.5) are moved by
    1e-12, which changes the result by less than 1e-12 because Phi_2 is
    continuous.
    '''
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    h = np.where(h == 0, 1e-12, h)
    k = np.where(k == 0, 1e-12, k)
    sq = np.sqrt(1 - rho ** 2)
    a_h = (k - rho * h) / (h * sq)
    a_k = (h - rho * k) / (k * sq)
    c = np.where(h * k > 0, 0.0, 0.5)
    return 0.5 * (norm.cdf(h) + norm.cdf(k)) - owens_t(h, a_h) - owens_t(k, a_k) - c


def rotated_tau(tau, p, rho, eps=1e-6):
    '''Observation-specific quantile level G = C(tau, p; rho) / p.

    MATLAB: G = copulacdf('Gaussian', [tau*ones(N1,1) pZ1], rho) ./ pZ1;

    For extreme rho and small p, G can come out as (numerically) 0 or 1. The
    interior point solver needs 0 < tau < 1 for every observation (its
    starting point is x = 1 - tau, strictly inside (0, 1)), so G is kept in
    [eps, 1 - eps]. The MATLAB code passes such values on silently.
    '''
    G = gaussian_copula_cdf(np.full(len(p), tau), p, rho) / p
    return np.clip(G, eps, 1 - eps)


def gaussian_copula_cdf(u, v, rho):
    '''C(u, v; rho) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho).
    MATLAB: copulacdf('Gaussian', [u v], rho)'''
    return bivariate_normal_cdf(norm.ppf(u), norm.ppf(v), rho)


# =============================================================================
# The estimator
# =============================================================================

def selection_qr(D, Y, X, B, vectau=(0.2, 0.4, 0.6, 0.8),
                 vecrho=np.round(np.arange(-0.98, 0.98 + 1e-9, 0.02), 2),
                 taus=np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2), verbose=False):
    '''Selection-corrected quantile regression (Code_QRS.m).

    Returns
    -------
    dict with
        rho : estimated copula parameter;
        beta : (1 + dim X) x len(taus) array of quantile coefficients
               (first row = intercept; columns = taus);
        taus, objective (the criterion over the rho grid), vecrho,
        gamma (probit coefficients), pZ (propensity scores).
    '''
    D = np.asarray(D).astype(int)
    X = np.atleast_2d(np.asarray(X, dtype=float).T).T     # n x k, even if X is 1-D
    B = np.atleast_2d(np.asarray(B, dtype=float).T).T
    Y = np.asarray(Y, dtype=float)

    # Estimate propensity score
    # MATLAB: Z=[X B];  gamma=glmfit(...);  pZ=normcdf(gamma(1)+Z*gamma(2:end));
    Z = np.hstack([X, B])                     # np.hstack takes ONE tuple of arrays
    gamma = probit(D, Z)
    pZ = norm.cdf(gamma[0] + Z @ gamma[1:])

    # Instrument function
    varphi = pZ

    # Select participants
    sel = D == 1
    Y1, pZ1, X1, varphi1 = Y[sel], pZ[sel], X[sel], varphi[sel]
    N1 = len(Y1)
    X1c = np.column_stack([np.ones(N1), X1])  # MATLAB: [ones(N1,1) X1]

    # Objective function over the rho grid
    vecrho = np.asarray(vecrho, dtype=float)
    obj_rho = np.zeros(len(vecrho))
    for j, rhoa in enumerate(vecrho):
        obj = 0.0
        for tau in vectau:
            # Gaussian copula: G = C(tau, p(Z); rho) / p(Z), the rank of the
            # tau-th quantile among participants ("rotated" quantile level)
            G = rotated_tau(tau, pZ1, rhoa)
            # Rotated quantile regression
            beta = rq(X1c, Y1, G)
            obj += np.mean(varphi1 * ((Y1 <= X1c @ beta) - G))
        obj_rho[j] = obj ** 2
        if verbose:
            print(f'rho = {rhoa:+.2f}: objective {obj_rho[j]:.3e}')

    # Minimize the objective function
    rho = vecrho[np.argmin(obj_rho)]

    # Estimate selection-corrected quantile parameters beta(tau)
    taus = np.asarray(taus, dtype=float)
    beta = np.zeros((X1c.shape[1], len(taus)))
    for j, tau in enumerate(taus):
        G = rotated_tau(tau, pZ1, rho)
        beta[:, j] = rq(X1c, Y1, G)

    return {'rho': float(rho), 'beta': beta, 'taus': taus, 'objective': obj_rho,
            'vecrho': vecrho, 'gamma': gamma, 'pZ': pZ}
