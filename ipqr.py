'''ipqr.py -- Nonlinear quantile regression by Koenker and Park's interior
point algorithm.

A Python port of ipqr.m and ipqr_objfunc.m, which implement

    Koenker, R. and B. J. Park (1996), "An Interior Point Algorithm for
    Nonlinear Quantile Regression," Journal of Econometrics 71, 265-283.

The MATLAB version passes the model as the NAME of a function (func) and
finds its Jacobian by prepending 'd' to that name (dfunc = ['d' func]), then
calls both through feval. In Python, functions are objects, so we simply pass
the two functions themselves. The MATLAB code also used flops() (removed from
MATLAB long ago) and fmin() (replaced by fminbnd); the line search here uses
scipy.optimize.minimize_scalar on the same interval [-1, 1].

Copyright (c) 2026 Eliot R. Weinstein. Licensed under the BSD 3-Clause
License; see LICENSE and NOTICE.md (the license covers this port only, not
the original algorithm or MATLAB code).
'''

import numpy as np
from scipy.optimize import minimize_scalar


def qr_objective(res, q):
    '''sum of rho_q(res): q * sum(res) - sum(res[res < 0]) (ipqr_objfunc.m).'''
    return q * np.sum(res) - np.sum(res[res < 0])


def ipqr(q, y, x, func, dfunc, beta, toler=1e-8, eta=0.97, inner=2, max_iter=500):
    '''Nonlinear regression quantile by the interior point algorithm.

    Parameters
    ----------
    q : the quantile, in (0, 1).
    y : (n,) response.
    x : predictors, in whatever form func and dfunc expect (often an n x m array).
    func : function (x, beta) -> (n,) fitted values f(x, beta).
        For linear quantile regression, func = lambda x, b: x @ b.
    dfunc : function (x, beta) -> (n, p) Jacobian of f with respect to beta.
        For linear quantile regression, dfunc = lambda x, b: x.
    beta : (p,) starting value.
    toler : minimum change in the objective before convergence is declared.
    eta : step-length factor (0.97 is the value recommended by Koenker and Park).
    inner : number of inner iterations per outer iteration (K&P recommend 2).
    max_iter : safeguard on the number of outer iterations (not in the MATLAB code).

    Returns
    -------
    dict with beta, objective, and iterations.

    The redundant input p of the MATLAB version (the length of beta) is
    dropped.
    '''
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float).copy()
    if not 0 < q < 1:
        raise ValueError('q should be strictly between 0 and 1.')
    n = len(y)
    d = np.zeros(n)
    residuals = y - func(x, beta)
    lastobj = qr_objective(residuals, q)
    change = np.inf
    iteration = 0

    while abs(change) > toler and iteration < max_iter:
        iteration += 1
        J = dfunc(x, beta)                             # J is the first-derivative matrix
        # MATLAB: dhat = d - J*(inv(J'*J)*(J'*d));   projection of d off the columns of J
        dhat = d - J @ np.linalg.solve(J.T @ J, J.T @ d)
        # MATLAB: m = toler + max([-dhat./(1-q); dhat./q]);
        m = toler + max(np.max(-dhat / (1 - q)), np.max(dhat / q))
        d = dhat / m

        # Inner iterations (K&P recommend two per outer iteration).
        for _ in range(inner):
            # MATLAB: dmatrix = min([q-d 1-q+d]')';  (row-wise minimum of the two columns)
            dmat = np.minimum(q - d, 1 - q + d)
            d2 = dmat ** 2
            # MATLAB: direc = inv(J'*(d2(:,ones(p,1)).*J))*(J'*(d2.*residuals));
            direc = np.linalg.solve(J.T @ (d2[:, None] * J), J.T @ (d2 * residuals))
            s = d2 * (residuals - J @ direc)
            alpha = max(np.max(s / (q - d)), np.max(-s / (1 - q + d)))
            d = d + eta / alpha * s

        # Line search recommended by K&P, on t in [-1, 1] along direc.
        # MATLAB: step = fmin('ipqr_objfunc', -1, 1, [], func, q, y, x, beta, direc) * direc;
        t = minimize_scalar(lambda t: qr_objective(y - func(x, beta + t * direc), q),
                            bounds=(-1, 1), method='bounded').x
        beta = beta + t * direc
        residuals = y - func(x, beta)
        obj = qr_objective(residuals, q)
        change = lastobj - obj
        lastobj = obj

    return {'beta': beta, 'objective': float(lastobj), 'iterations': iteration}
