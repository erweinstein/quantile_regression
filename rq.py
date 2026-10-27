'''rq.py -- Quantile regression by the Frisch-Newton interior point method.

A Python port of rq.m (the function rq_fnm and its helpers lp_fnm and bound),
the basic version of the interior point (Frisch-Newton) algorithm for quantile
regression developed for Roger Koenker's R package quantreg:

    Function lp_fnm of Daniel Morillo & Roger Koenker
    Translated from Ox to Matlab by Paul Eilers 1999
    Modified by Roger Koenker 2000--
    More changes by Paul Eilers 2004
    http://www.econ.uiuc.edu/~roger/research/rq/rq.html

and of rqic.m, the version of the same algorithm that allows linear inequality
constraints R b >= r (Roger Koenker, last revised April 20 2002).

Reference: Portnoy, S. and R. Koenker (1997), "The Gaussian Hare and the
Laplacian Tortoise: Computability of Squared-Error versus Absolute-Error
Estimators," Statistical Science 12(4), 279-300.

Originally ported for MSBI 31600 (Summer 2020) by Eliot R. Weinstein;
cleaned up and completed in 2026 (see README.md for what changed and why).

Copyright (c) 2020, 2026 Eliot R. Weinstein. Licensed under the BSD 3-Clause
License; see LICENSE and NOTICE.md (the license covers this port only, not the
original algorithms or MATLAB code of Koenker, Morillo, Eilers, Arellano, and
Bonhomme).

How the quantile regression problem becomes a linear program
-------------------------------------------------------------
The tau-th regression quantile minimizes sum_i rho_tau(y_i - x_i' b), with
rho_tau(u) = u (tau - 1{u < 0}). Its dual is the bounded-variable linear
program

    max_a { y' a  :  X' a = (1 - tau) X' 1,  0 <= a <= 1 },

and the regression coefficients are (minus) the Lagrange multipliers of the
equality constraints. rq_fnm sets up this dual and lp_fnm solves it.

A note on shapes. MATLAB code like this juggles row vectors (1 x n) and column
vectors (n x 1) and transposes between them constantly (y', c', dy', ...).
numpy's 1-D arrays have no orientation, so every vector here is 1-D and the
transposes simply disappear; matrices stay 2-D. The comments show the MATLAB
line next to its translation where the correspondence is not obvious.
'''

import warnings

import numpy as np


# =============================================================================
# Helper used by both solvers
# =============================================================================

def bound(x, dx):
    '''Allowed step lengths for the interior point iterations: for each
    element with dx < 0, the step -x/dx that would take x to zero; 1e20
    (effectively infinity) elsewhere.

    MATLAB:  b = 1e20 + 0 * x;  f = find(dx < 0);  b(f) = -x(f) ./ dx(f);

    In numpy a boolean mask does the job of find(): it selects the same
    elements, and there is no 1-based vs 0-based index to convert because we
    never look at the index numbers themselves. (My 2020 version subtracted 1
    from the indices returned by .nonzero(), which are already 0-based; that
    was a bug.)
    '''
    b = np.full_like(x, 1e20, dtype=float)
    neg = dx < 0
    b[neg] = -x[neg] / dx[neg]
    return b


def _step_lengths(beta, *pairs):
    '''min(beta * min over all (x, dx) pairs of bound(x, dx), 1), i.e. the
    MATLAB lines  fp = min(fx, fs);  fp = min(min(beta * fp), 1);'''
    return min(beta * min(np.min(bound(v, dv)) for v, dv in pairs), 1.0)


# =============================================================================
# lp_fnm: bounded-variable linear program by the Frisch-Newton method
# =============================================================================

def lp_fnm(A, c, b, u, x, beta=0.9995, small=1e-5, max_it=50):
    '''Solve a linear program by the interior point method:

        min(c' x)  s.t.  A x = b  and  0 < x < u,

    following Mehrotra's predictor-corrector steps. An initial feasible
    solution has to be provided as x.

    Parameters
    ----------
    A : (p, n) array.
    c, u, x : (n,) arrays.
    b : (p,) array.
    beta, small, max_it : step-length damping, convergence tolerance on the
        duality gap, and maximum number of iterations. As the original
        authors note, you can modify these constants if you want to.

    Returns
    -------
    y : (p,) array, the dual solution (Lagrange multipliers of A x = b).
    '''
    A = np.asarray(A, dtype=float)
    c, b, u, x = (np.asarray(v, dtype=float).copy() for v in (c, b, u, x))
    p, n = A.shape

    # Generate initial feasible point
    s = u - x
    # MATLAB: y = (A' \ c')';  -- least squares, since A' is n x p (not square)
    y = np.linalg.lstsq(A.T, c, rcond=None)[0]
    # MATLAB: r = c - y * A;
    r = c - A.T @ y
    # MATLAB: r = r + 0.001 * (r == 0);   (PE 2004)
    # (r == 0) is an elementwise 0/1 indicator. My 2020 version used
    # `if r == 0:`, which compares a whole array and does not work in numpy;
    # np.where keeps the elementwise meaning.
    r = r + 0.001 * (r == 0)
    # MATLAB: z = r .* (r > 0);   i.e. the positive part of r
    z = np.where(r > 0, r, 0.0)
    w = z - r
    gap = c @ x - y @ b + w @ u

    # Start iterations
    it = 0
    while gap > small and it < max_it:
        it += 1

        # Compute affine step
        q = 1.0 / (z / x + w / s)
        r = z - w
        # MATLAB: Q = spdiags(sqrt(q), 0, n, n); AQ = A * Q; rhs = Q * r';
        # Multiplying by a diagonal matrix just rescales columns (or
        # elements), so there is no need to build the sparse matrix at all.
        sq = np.sqrt(q)
        AQ = A * sq[None, :]
        rhs = sq * r
        dy = np.linalg.lstsq(AQ.T, rhs, rcond=None)[0]      # MATLAB: dy = (AQ' \ rhs)';
        dx = q * (A.T @ dy - r)                              # MATLAB: dx = q .* (dy * A - r)';
        ds = -dx
        dz = -z * (1 + dx / x)
        dw = -w * (1 + ds / s)

        # Compute maximum allowable step lengths
        fp = _step_lengths(beta, (x, dx), (s, ds))
        fd = _step_lengths(beta, (w, dw), (z, dz))

        # If full step is feasible, take it. Otherwise modify it
        if min(fp, fd) < 1:

            # Update mu
            mu = z @ x + w @ s
            g = (z + fd * dz) @ (x + fp * dx) + (w + fd * dw) @ (s + fp * ds)
            # MATLAB: mu = mu * (g / mu) ^3 / ( 2* n);
            # (In Python ^ is bitwise XOR; the power operator is **.)
            mu = mu * (g / mu) ** 3 / (2 * n)

            # Compute modified step
            dxdz = dx * dz
            dsdw = ds * dw
            xinv = 1.0 / x
            sinv = 1.0 / s
            xi = mu * (xinv - sinv)
            rhs = rhs + sq * (dxdz - dsdw - xi)
            dy = np.linalg.lstsq(AQ.T, rhs, rcond=None)[0]
            dx = q * (A.T @ dy + xi - r - dxdz + dsdw)
            ds = -dx
            dz = mu * xinv - z - xinv * z * dx - dxdz
            dw = mu * sinv - w - sinv * w * ds - dsdw

            # Compute maximum allowable step lengths
            fp = _step_lengths(beta, (x, dx), (s, ds))
            fd = _step_lengths(beta, (w, dw), (z, dz))

        # Take the step. (This happens on every iteration, whether or not the
        # step was modified above; in my 2020 version it sat in an `else:`
        # branch, and `return` was inside the loop, so the algorithm stopped
        # after one iteration.)
        x = x + fp * dx
        s = s + fp * ds
        y = y + fd * dy
        w = w + fd * dw
        z = z + fd * dz
        gap = c @ x - y @ b + w @ u

    if gap > small:
        warnings.warn(f'lp_fnm: max_it = {max_it} reached before convergence (gap = {gap:.2e})')
    return y


def rq(X, y, tau=0.5, **kwargs):
    '''Regression quantile(s) by the Frisch-Newton interior point method
    (rq_fnm in rq.m).

    Constructs the dual problem of quantile regression and solves it with
    lp_fnm.

    Parameters
    ----------
    X : (n, p) design matrix. Include a column of ones for an intercept;
        rq() does not add one (as in the MATLAB original).
    y : (n,) response.
    tau : the quantile, a number in (0, 1). It may also be an (n,) array of
        observation-specific "quantiles", as used by the rotated quantile
        regression of Arellano and Bonhomme (see selection_qr.py).
    **kwargs : beta, small, max_it, passed to lp_fnm.

    Returns
    -------
    (p,) array of coefficients.
    '''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError('X should be a 2-D (n x p) design matrix.')
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f'y should have length n = {n}.')
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (n,))
    if np.any((tau <= 0) | (tau >= 1)):
        raise ValueError('tau should be strictly between 0 and 1.')
    if np.linalg.matrix_rank(X) < p:
        raise ValueError('X does not have full column rank.')

    # MATLAB:  u = ones(m, 1);  a = (1 - p) .* u;  b = -lp_fnm(X', -y', X' * a, u, a)';
    u = np.ones(n)
    a = (1 - tau) * u
    return -lp_fnm(X.T, -y, X.T @ a, u, a, **kwargs)


# =============================================================================
# rqic: quantile regression with linear inequality constraints R b >= r
# =============================================================================

def lp_fnm_ineq(A1, c1, A2, c2, b, u, x1, x2, beta=0.9995, small=1e-8, max_it=100):
    '''Solve a linear program by the interior point method (lp_fnm in rqic.m):

        min{ c1' x1 + c2' x2  |  A1 x1 + A2 x2 = b,  0 < x1 < u,  0 < x2 }.

    An initial (possibly infeasible) solution has to be provided as
    (x1, x2): it should satisfy 0 < x1 < u and 0 < x2 but need not satisfy
    A x = b.

    History (from Koenker's original comments): based on the Lustig, Marsden
    and Shanno (1992) description of an implementation of Mehrotra's (1990)
    predictor-corrector interior point method for bounded variable linear
    programming; loosely based on the S-Plus and Fortran versions of the
    algorithm by Roger Koenker described in Portnoy and Koenker (1997).
    '''
    A1, A2 = np.asarray(A1, dtype=float), np.asarray(A2, dtype=float)
    c1, c2, b, u, x1, x2 = (np.asarray(v, dtype=float).copy() for v in (c1, c2, b, u, x1, x2))
    n1, n2 = A1.shape[1], A2.shape[1]

    # Generate an initial point
    s = u - x1
    y = np.linalg.lstsq(A1.T, c1, rcond=None)[0]
    r1 = c1 - A1.T @ y
    z1 = np.where(r1 > 0, r1, 0.0)
    w = z1 - r1
    z2 = np.ones(n2)
    gap = z1 @ x1 + z2 @ x2 + w @ s

    it = 0
    while gap > small and it < max_it:
        it += 1

        # Compute affine step
        r1 = c1 - A1.T @ y
        r2 = c2 - A2.T @ y
        r3 = b - A1 @ x1 - A2 @ x2
        q1 = 1.0 / (z1 / x1 + w / s)
        q2 = x2 / z2
        AQ1 = A1 * q1[None, :]
        AQ2 = A2 * q2[None, :]
        AQA = AQ1 @ A1.T + AQ2 @ A2.T
        rhs = r3 + AQ1 @ r1 + AQ2 @ r2
        dy = np.linalg.solve(AQA, rhs)
        dx1 = q1 * (A1.T @ dy - r1)
        dx2 = q2 * (A2.T @ dy - r2)
        ds = -dx1
        dz1 = -z1 * (1 + dx1 / x1)
        dz2 = -z2 * (1 + dx2 / x2)
        dw = -w * (1 + ds / s)

        # Compute maximum allowable step lengths
        fp = _step_lengths(beta, (x1, dx1), (s, ds), (x2, dx2))
        fd = _step_lengths(beta, (w, dw), (z1, dz1), (z2, dz2))

        # If full affine step is feasible, take it. Otherwise modify it
        if min(fp, fd) < 1:
            mu = z1 @ x1 + z2 @ x2 + w @ s
            g = ((z1 + fd * dz1) @ (x1 + fp * dx1) + (z2 + fd * dz2) @ (x2 + fp * dx2)
                 + (w + fd * dw) @ (s + fp * ds))
            # MATLAB: mu = mu * (g / mu) ^3 / ( 2* n1 + n2);
            mu = mu * (g / mu) ** 3 / (2 * n1 + n2)

            # Compute modified step
            dxdz1 = dx1 * dz1
            dxdz2 = dx2 * dz2
            dsdw = ds * dw
            xinv1 = 1.0 / x1
            xinv2 = 1.0 / x2
            sinv = 1.0 / s
            xi1 = xinv1 * dxdz1 - sinv * dsdw - mu * (xinv1 - sinv)
            xi2 = xinv2 * dxdz2 - mu * xinv2
            rhs = rhs + A1 @ (q1 * xi1) + A2 @ (q2 * xi2)
            dy = np.linalg.solve(AQA, rhs)
            dx1 = q1 * (A1.T @ dy - r1 - xi1)
            dx2 = q2 * (A2.T @ dy - r2 - xi2)
            ds = -dx1
            dz1 = -z1 + xinv1 * (mu - z1 * dx1 - dxdz1)
            dz2 = -z2 + xinv2 * (mu - z2 * dx2 - dxdz2)
            dw = -w + sinv * (mu - w * ds - dsdw)

            fp = _step_lengths(beta, (x1, dx1), (s, ds), (x2, dx2))
            fd = _step_lengths(beta, (w, dw), (z1, dz1), (z2, dz2))

        # Take the step
        x1 = x1 + fp * dx1
        x2 = x2 + fp * dx2
        z1 = z1 + fd * dz1
        z2 = z2 + fd * dz2
        s = s + fp * ds
        y = y + fd * dy
        w = w + fd * dw
        gap = z1 @ x1 + z2 @ x2 + w @ s

    if gap > small:
        warnings.warn('max_it exceeded: non-convergence')
    return y


def rqic(X, y, R, r, tau=0.5, **kwargs):
    '''Regression quantile subject to linear inequality constraints R b >= r
    (rqic.m).

    Parameters
    ----------
    X : (n, p) design matrix (include a column of ones for an intercept).
    y : (n,) response.
    R : (m, p) constraint matrix; r : (m,) right-hand side.
    tau : quantile in (0, 1).

    Example: to force the slope (second coefficient) to be nonnegative,
    R = [[0, 1]], r = [0].
    '''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n, p = X.shape
    m = R.shape[0]
    if R.shape[1] != p or r.shape != (m,):
        raise ValueError('R should be m x p and r should have length m.')
    # MATLAB: u = ones(n,1); a1 = (1-tau)*u; a2 = ones(m,1); b = X' * a1;
    #         beta = -lp_fnm(X', -y', R', -r', b, u, a1, a2)';
    u = np.ones(n)
    a1 = (1 - tau) * u
    a2 = np.ones(m)
    return -lp_fnm_ineq(X.T, -y, R.T, -r, X.T @ a1, u, a1, a2, **kwargs)


def rq_objective(X, y, b, tau):
    '''The quantile regression objective sum_i rho_tau(y_i - x_i' b), useful
    for checking solutions (and for Koenker's pseudo-R^2; see the FAQ).'''
    e = np.asarray(y) - np.asarray(X) @ np.asarray(b)
    return float(np.sum(e * (tau - (e < 0))))
