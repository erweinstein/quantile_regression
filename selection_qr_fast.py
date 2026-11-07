'''selection_qr_fast.py -- faster computation of the same estimators as rq.rq() and
selection_qr.selection_qr(), which are left unchanged (they replicate rq.m and
Code_QRS.m line by line).

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com

Three changes, each meant to return the same estimate, only faster:

1. lp_fnm_fast(): the same Frisch-Newton (Mehrotra predictor-corrector) iterations as
   rq.lp_fnm(), with the step-length bounds computed in one vectorized pass and the
   p x p normal equations solved by Cholesky instead of a least-squares solve of an
   n x p system. Same iterates up to rounding.

2. rq_preprocessed(): Portnoy and Koenker's (1997) preprocessing, written for
   observation-specific quantile levels tau_i (the "rotated" quantile regression of
   Arellano and Bonhomme). Given a preliminary estimate b0 (here the solution at the
   neighbouring copula value), only the M observations with the smallest residuals are
   kept; the others are summarized by two pseudo-observations ("globs"), one for those
   expected above the fitted line and one for those below. The reduced problem is
   solved, and the signs of the globbed observations' residuals are checked. If any
   sign is wrong, those observations are added back and the problem is solved again.
   When every sign is right, the reduced linear program has the same solution as the
   full one, so the estimate is the full-sample estimate (to the solver's tolerance).
   One caveat for the selection estimator: p observations lie exactly on each fitted
   line, and in floating point they land slightly above or below it depending on the
   path the solver took. Code_QRS.m's moment 1{Y <= X'b} counts them by that rounding,
   so preprocessing can move the estimated copula parameter by a grid step or two even
   though every regression reaches the same optimum. It is therefore an option.

3. selection_qr_fast(): Arellano and Bonhomme's (2017) estimator computed with (1) and,
   optionally, (2), sweeping the copula grid in order so each fit starts from the previous one
   (the warm starts of Pereda-Fernandez 2024, Algorithm 1). Optionally (search =
   "coarse"), the copula objective is first evaluated on every k-th grid point and then
   on the full grid only around the n_best best coarse values (in the spirit of
   Pereda-Fernandez's Algorithm 3); this is exact when the overall minimizer lies in
   one of those neighbourhoods, which the function reports it cannot guarantee.
   settings = "paper" uses the paper's choices for the copula search (the nine deciles,
   and 200 grid points; Arellano and Bonhomme 2017, section 5) instead of
   Code_QRS.m's (four quantiles 0.2-0.8, 99 points).

References
- Arellano M, Bonhomme S (2017). Quantile selection models with an application to
  understanding changes in wage inequality. Econometrica 85(1), 1-28.
- Pereda-Fernandez S (2024). Fast algorithms for quantile regression with selection.
  arXiv:2402.16693.
- Portnoy S, Koenker R (1997). The Gaussian hare and the Laplacian tortoise:
  computability of squared-error versus absolute-error estimators. Statistical
  Science 12(4), 279-300.
'''

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import norm

from selection_qr import probit, rotated_tau


# =============================================================================
# 1. A vectorized Frisch-Newton solver
# =============================================================================

def _max_step(beta, *pairs):
    m = np.inf
    for v, dv in pairs:
        neg = dv < 0
        if neg.any():
            m = min(m, np.min(-v[neg] / dv[neg]))
    return min(beta * m, 1.0)


def _solve_normal(AQ, rhs):
    '''Least-squares solution of AQ' dy = rhs via the p x p normal equations.'''
    M = AQ @ AQ.T
    try:
        return cho_solve(cho_factor(M, check_finite=False), AQ @ rhs, check_finite=False)
    except (LinAlgError, ValueError):
        return np.linalg.lstsq(AQ.T, rhs, rcond=None)[0]


def lp_fnm_fast(A, c, b, u, x, beta=0.9995, small=1e-5, max_it=50):
    '''Same algorithm and arguments as rq.lp_fnm(); returns the dual solution y.'''
    A = np.asarray(A, dtype=float)
    c, b, u, x = (np.asarray(v, dtype=float).copy() for v in (c, b, u, x))
    n = A.shape[1]
    s = u - x
    y = np.linalg.lstsq(A.T, c, rcond=None)[0]
    r = c - A.T @ y
    r = r + 0.001 * (r == 0)
    z = np.where(r > 0, r, 0.0)
    w = z - r
    gap = c @ x - y @ b + w @ u
    it = 0
    while gap > small and it < max_it:
        it += 1
        q = 1.0 / (z / x + w / s)
        r = z - w
        sq = np.sqrt(q)
        AQ = A * sq[None, :]
        rhs = sq * r
        dy = _solve_normal(AQ, rhs)
        dx = q * (A.T @ dy - r)
        ds = -dx
        dz = -z * (1 + dx / x)
        dw = -w * (1 + ds / s)
        fp = _max_step(beta, (x, dx), (s, ds))
        fd = _max_step(beta, (w, dw), (z, dz))
        if min(fp, fd) < 1:
            mu = z @ x + w @ s
            g = (z + fd * dz) @ (x + fp * dx) + (w + fd * dw) @ (s + fp * ds)
            mu = mu * (g / mu) ** 3 / (2 * n)
            dxdz = dx * dz
            dsdw = ds * dw
            xinv = 1.0 / x
            sinv = 1.0 / s
            xi = mu * (xinv - sinv)
            rhs = rhs + sq * (dxdz - dsdw - xi)
            dy = _solve_normal(AQ, rhs)
            dx = q * (A.T @ dy + xi - r - dxdz + dsdw)
            ds = -dx
            dz = mu * xinv - z - xinv * z * dx - dxdz
            dw = mu * sinv - w - sinv * w * ds - dsdw
            fp = _max_step(beta, (x, dx), (s, ds))
            fd = _max_step(beta, (w, dw), (z, dz))
        x = x + fp * dx
        s = s + fp * ds
        y = y + fd * dy
        w = w + fd * dw
        z = z + fd * dz
        gap = c @ x - y @ b + w @ u
    if gap > small:
        warnings.warn(f'lp_fnm_fast: max_it = {max_it} reached before convergence (gap = {gap:.2e})')
    return y


def rq_fast(X, y, tau=0.5, **kwargs):
    '''Same as rq.rq() (tau a number or an (n,) array), with lp_fnm_fast().'''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (n,))
    a = 1 - tau
    return -lp_fnm_fast(X.T, -y, X.T @ a, np.ones(n), a, **kwargs)


def check_objective(X, y, tau, b):
    '''sum_i rho_{tau_i}(y_i - x_i'b).'''
    r = np.asarray(y) - np.asarray(X) @ b
    tau = np.broadcast_to(np.asarray(tau, dtype=float), r.shape)
    return float(np.sum(r * (tau - (r < 0))))


# =============================================================================
# 2. Preprocessing with globs, for observation-specific tau
# =============================================================================

def rq_preprocessed(X, y, tau, b0, m=None, max_rounds=8, theta=0.5):
    '''rq of y on X at levels tau (number or (n,) array), using b0 to set aside the
    observations whose residual signs are predictable. Returns (b, info) where info
    reports the rounds, the size of the final reduced problem and whether it fell back
    to the full problem.'''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (n,))
    if m is None:
        m = int(min(n, max(10 * p, np.ceil(3 * np.sqrt(n * p)))))
    if m >= n:
        return rq_fast(X, y, tau), {'rounds': 0, 'kept': n, 'fallback': True}
    r0 = y - X @ b0
    order = np.argsort(np.abs(r0))
    keep = np.zeros(n, dtype=bool)
    keep[order[:m]] = True
    scale = np.std(y) + 1.0
    for rnd in range(1, max_rounds + 1):
        up = ~keep & (r0 > 0)
        lo = ~keep & (r0 <= 0)
        Xr, yr, tr = [X[keep]], [y[keep]], [tau[keep]]
        big = 1e3 * scale * (1.0 + np.abs(X).sum(axis=0).max())
        if up.any():
            xg = (tau[up][:, None] * X[up]).sum(axis=0) / theta
            Xr.append(xg[None, :]); yr.append([xg @ b0 + big]); tr.append([theta])
        if lo.any():
            xg = ((1 - tau[lo])[:, None] * X[lo]).sum(axis=0) / (1 - theta)
            Xr.append(xg[None, :]); yr.append([xg @ b0 - big]); tr.append([theta])
        Xr = np.vstack(Xr); yr = np.concatenate([np.asarray(v, dtype=float) for v in yr])
        tr = np.concatenate([np.asarray(v, dtype=float) for v in tr])
        b = rq_fast(Xr, yr, tr)
        r = y - X @ b
        bad = (up & (r < 0)) | (lo & (r > 0))
        if not bad.any():
            return b, {'rounds': rnd, 'kept': int(keep.sum()), 'fallback': False}
        # add the wrongly signed observations and, as Portnoy and Koenker do, widen the band
        keep |= bad
        extra = order[~keep[order]][:m]
        keep[extra] = True
        if keep.sum() >= n:
            break
    return rq_fast(X, y, tau), {'rounds': max_rounds, 'kept': n, 'fallback': True}


# =============================================================================
# 3. The selection-corrected estimator
# =============================================================================

SETTINGS = {
    'code': {'vectau': (0.2, 0.4, 0.6, 0.8), 'vecrho': np.round(np.arange(-0.98, 0.98 + 1e-9, 0.02), 2)},
    # The paper (section 5) uses the nine deciles and 200 grid points; it does not give
    # the grid's end points here, so we space 200 points evenly over (-1, 1).
    'paper': {'vectau': tuple(np.round(np.arange(0.1, 0.9 + 1e-9, 0.1), 1)), 'vecrho': np.linspace(-0.995, 0.995, 200)},
}


def _objective_at(rhoa, vectau, X1c, Y1, pZ1, starts, preprocess, tie=0.0):
    obj = 0.0
    sols = []
    for k, tau in enumerate(vectau):
        G = rotated_tau(tau, pZ1, rhoa)
        if preprocess and starts[k] is not None:
            beta, _ = rq_preprocessed(X1c, Y1, G, starts[k])
        else:
            beta = rq_fast(X1c, Y1, G)
        sols.append(beta)
        obj += np.mean(pZ1 * ((Y1 <= X1c @ beta + tie) - G))
    return obj ** 2, sols


def selection_qr_fast(D, Y, X, B, settings='code', vectau=None, vecrho=None,
                      taus=np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2),
                      preprocess=False, search='full', coarse_step=5, n_best=3, tie_tol=0.0):
    '''Arellano and Bonhomme's estimator, as selection_qr.selection_qr(), computed faster.

    settings : "code" (Code_QRS.m's choices, the default, as selection_qr()) or "paper";
               vectau / vecrho, if given, override them.
    tie_tol  : residuals within tie_tol x (1 + sd of Y among participants) of zero count as
               Y <= X'b. At the solution, p observations lie exactly on the fitted line
               (the basis), but the interior point method leaves them at about +-1e-9,
               so whether they count in 1{Y <= X'b} is decided by rounding. With
               tie_tol = 0 (the default) this matches Code_QRS.m and selection_qr();
               tie_tol = 1e-7 counts them, as the definition (<=) says, and makes the
               result independent of the solver's rounding.
    preprocess : False (the default) or True. Without preprocessing the estimates are those of
               selection_qr() to rounding error; with it, the rotated regressions reach the
               same optima, but the p observations on each fitted line can fall on the
               other side of it in floating point, which can move rho by a grid step or
               two (in checks/fast_vs_original.py, in 16 of 30 data sets of 2,000 and 0 of
               10 of 8,000; in checks/timing.py, 1 of 3 of 8,000). That is far below
               rho's standard error, but it is not the same number, so it is off by default.
    search   : "full" evaluates the copula objective at every grid point (the same
               estimate as selection_qr() with the same grid); "coarse" evaluates every
               coarse_step-th point first and then the full grid around the n_best best.
    Returns the same dict as selection_qr() plus 'search_info'.
    '''
    vectau = SETTINGS[settings]['vectau'] if vectau is None else tuple(vectau)
    vecrho = SETTINGS[settings]['vecrho'] if vecrho is None else np.asarray(vecrho, dtype=float)
    D = np.asarray(D).astype(int)
    X = np.atleast_2d(np.asarray(X, dtype=float).T).T
    B = np.atleast_2d(np.asarray(B, dtype=float).T).T
    Y = np.asarray(Y, dtype=float)
    Z = np.hstack([X, B])
    gamma = probit(D, Z)
    pZ = norm.cdf(gamma[0] + Z @ gamma[1:])
    sel = D == 1
    Y1, pZ1, X1 = Y[sel], pZ[sel], X[sel]
    X1c = np.column_stack([np.ones(len(Y1)), X1])

    tie = tie_tol * (1.0 + np.std(Y1))
    J = len(vecrho)
    obj_rho = np.full(J, np.nan)
    if search == 'full':
        todo = [list(range(J))]
    elif search == 'coarse':
        todo = [list(range(0, J, coarse_step)) + ([J - 1] if (J - 1) % coarse_step else [])]
    else:
        raise ValueError('search should be "full" or "coarse".')

    def sweep(idx):
        starts = [None] * len(vectau)
        prev = None
        for j in sorted(idx):
            if not np.isnan(obj_rho[j]):
                continue
            if prev is None or j - prev > coarse_step:      # not adjacent: no warm start
                starts = [None] * len(vectau)
            obj_rho[j], starts = _objective_at(vecrho[j], vectau, X1c, Y1, pZ1, starts, preprocess, tie)
            prev = j

    sweep(todo[0])
    info = {'evaluated_coarse': int(np.sum(~np.isnan(obj_rho)))}
    if search == 'coarse':
        coarse = np.array(todo[0])
        best = coarse[np.argsort(obj_rho[coarse])[:n_best]]
        fine = sorted({k for j in best for k in range(max(0, j - coarse_step + 1), min(J, j + coarse_step))})
        sweep(fine)
        info['evaluated_total'] = int(np.sum(~np.isnan(obj_rho)))
        info['note'] = ('exact if the full-grid minimizer lies within coarse_step - 1 grid points of one of '
                        'the n_best best coarse points; not guaranteed')
    rho = vecrho[np.nanargmin(obj_rho)]

    taus = np.asarray(taus, dtype=float)
    beta = np.zeros((X1c.shape[1], len(taus)))
    prev = None
    for j, tau in enumerate(taus):
        G = rotated_tau(tau, pZ1, rho)
        if preprocess and prev is not None:
            beta[:, j], _ = rq_preprocessed(X1c, Y1, G, prev)
        else:
            beta[:, j] = rq_fast(X1c, Y1, G)
        prev = beta[:, j]
    return {'rho': float(rho), 'beta': beta, 'taus': taus, 'objective': obj_rho, 'vecrho': vecrho,
            'gamma': gamma, 'pZ': pZ, 'search_info': info}
