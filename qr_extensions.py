'''qr_extensions.py -- additions to the ports in rq.py and selection_qr.py, which are
left unchanged so that they keep replicating rq.m, rqic.m and Code_QRS.m.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com

What is added, and where it comes from:

1. rq_se(): standard errors for a regression quantile. "ker" is Powell's (1991) kernel
   sandwich as computed by quantreg's summary.rq(se = "ker") (Gaussian kernel; bandwidth
   from Hall and Sheather (1988), scaled by a robust residual scale); "iid" is Koenker and
   Bassett's (1978) formula with the sparsity estimated by a difference quotient of
   residual quantiles (Siddiqui 1960), a simpler version of quantreg's se = "iid".
   Neither rq.m nor Code_QRS.m reports standard errors.

2. weighted_rq(): a regression quantile with nonnegative observation weights. The check
   function is positively homogeneous, w * rho_tau(u) = rho_tau(w * u) for w >= 0, so
   multiplying the rows of X and y by w gives the weighted problem with the same solver.

3. selection_qr_weighted() and selection_qr_bootstrap(): Arellano and Bonhomme's (2017)
   estimator with observation weights in every step (the probit, the moment condition
   for rho and the rotated quantile regressions), and the weighted (multiplier)
   bootstrap built on it, with Exp(1) weights (Ma and Kosorok 2005), the inference
   Pereda-Fernandez (2024) recommends for this estimator, whose asymptotic variance is
   complicated. Arellano and Bonhomme report subsampling intervals in their
   application.

4. rearrange(): Chernozhukov, Fernandez-Val and Galichon's (2010) rearrangement. An
   estimated quantile process x'beta(tau) need not increase in tau (quantile
   crossing); sorting its values over the tau grid gives a monotone estimate that is
   never farther from a monotone truth, in any L_p norm (their Proposition 1).

References
- Arellano M, Bonhomme S (2017). Quantile selection models with an application to
  understanding changes in wage inequality. Econometrica 85(1), 1-28.
- Chernozhukov V, Fernandez-Val I, Galichon A (2010). Quantile and probability curves
  without crossing. Econometrica 78(3), 1093-1125.
- Hall P, Sheather SJ (1988). On the distribution of a studentized quantile. JRSS B
  50(3), 381-391.
- Koenker R (2005). Quantile Regression. Cambridge University Press, section 3.4.
- Koenker R, Bassett G (1978). Regression quantiles. Econometrica 46(1), 33-50.
- Ma S, Kosorok MR (2005). Robust semiparametric M-estimation and the weighted
  bootstrap. Journal of Multivariate Analysis 96(1), 190-217.
- Pereda-Fernandez S (2024). Fast algorithms for quantile regression with selection.
  arXiv:2402.16693.
- Siddiqui MM (1960). Distribution of quantiles in samples from a bivariate population.
  Journal of Research of the National Bureau of Standards B 64(3), 145-150.
- Powell JL (1991). Estimation of monotonic regression models under quantile
  restrictions. In Barnett, Powell and Tauchen (eds), Nonparametric and Semiparametric
  Methods in Econometrics and Statistics, Cambridge University Press, 357-384.
'''

import numpy as np
from scipy.stats import norm

from rq import rq
from selection_qr import rotated_tau


# =============================================================================
# Standard errors for rq()
# =============================================================================

def hall_sheather(n, tau, alpha=0.05):
    '''Hall-Sheather (1988) bandwidth on the probability scale, as in quantreg's
    bandwidth.rq(hs = TRUE).'''
    x0 = norm.ppf(tau)
    f0 = norm.pdf(x0)
    return n ** (-1 / 3) * norm.ppf(1 - alpha / 2) ** (2 / 3) * (
        (1.5 * f0 ** 2) / (2 * x0 ** 2 + 1)) ** (1 / 3)


def rq_se(X, y, tau=0.5, beta=None, method='ker', alpha=0.05):
    '''Standard errors of rq(X, y, tau).

    method = "iid": Koenker and Bassett's tau(1 - tau) s^2 (X'X)^{-1}, with the sparsity
             s = 1/f(F^{-1}(tau)) estimated by a difference quotient of the residual
             quantiles over tau +- h (h from Hall and Sheather).
    method = "ker": Powell's sandwich tau(1 - tau) H^{-1} (X'X / n) H^{-1} / n, with
             H = sum_i phi(u_i / k) / k x_i x_i' / n and
             k = (Phi^{-1}(tau + h) - Phi^{-1}(tau - h)) min(sd(u), IQR(u) / 1.34),
             as in quantreg's summary.rq(se = "ker").
    Returns (se, cov).
    '''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if beta is None:
        beta = rq(X, y, tau)
    u = y - X @ beta
    h = hall_sheather(n, tau, alpha)
    lo, hi = max(tau - h, 1e-6), min(tau + h, 1 - 1e-6)
    XtX = X.T @ X / n
    if method == 'iid':
        s = (np.quantile(u, hi) - np.quantile(u, lo)) / (hi - lo)
        cov = tau * (1 - tau) * s ** 2 * np.linalg.inv(XtX) / n
    elif method == 'ker':
        q75, q25 = np.quantile(u, [0.75, 0.25])
        k = (norm.ppf(tau + h) - norm.ppf(tau - h)) * min(np.std(u, ddof=1), (q75 - q25) / 1.34)
        f = norm.pdf(u / k) / k
        H = (X * f[:, None]).T @ X / n
        Hi = np.linalg.inv(H)
        cov = tau * (1 - tau) * Hi @ XtX @ Hi / n
    else:
        raise ValueError('method should be "iid" or "ker".')
    return np.sqrt(np.diag(cov)), cov


# =============================================================================
# Weighted regression quantiles and the weighted selection-corrected estimator
# =============================================================================

def weighted_rq(X, y, tau, w):
    '''argmin_b sum_i w_i rho_tau(y_i - x_i'b) for w_i >= 0 (tau may be an (n,) array).
    Observations with w_i = 0 are dropped.'''
    w = np.asarray(w, dtype=float)
    keep = w > 0
    tau = np.broadcast_to(np.asarray(tau, dtype=float), w.shape)[keep]
    return rq(np.asarray(X, dtype=float)[keep] * w[keep, None], np.asarray(y, dtype=float)[keep] * w[keep],
              tau if tau.size > 1 else float(tau))


def weighted_probit(D, Z, w, tol=1e-10, max_iter=100):
    '''Probit with observation weights (Fisher scoring), constant added; with w = 1 it
    is selection_qr.probit().'''
    D = np.asarray(D, dtype=float)
    w = np.asarray(w, dtype=float)
    W = np.column_stack([np.ones(len(D)), np.asarray(Z, dtype=float)])
    g = np.zeros(W.shape[1])
    for _ in range(max_iter):
        xb = W @ g
        P = np.clip(norm.cdf(xb), 1e-12, 1 - 1e-12)
        phi = norm.pdf(xb)
        score = W.T @ (w * phi * (D - P) / (P * (1 - P)))
        H = (W * (w * phi ** 2 / (P * (1 - P)))[:, None]).T @ W
        step = np.linalg.solve(H, score)
        g = g + step
        if np.max(np.abs(step)) < tol:
            break
    return g


def selection_qr_weighted(D, Y, X, B, w=None, vectau=(0.2, 0.4, 0.6, 0.8),
                          vecrho=np.round(np.arange(-0.98, 0.98 + 1e-9, 0.02), 2),
                          taus=np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)):
    '''selection_qr() with observation weights w in every step (w = None: unweighted,
    which reproduces selection_qr()).'''
    D = np.asarray(D).astype(int)
    X = np.atleast_2d(np.asarray(X, dtype=float).T).T
    B = np.atleast_2d(np.asarray(B, dtype=float).T).T
    Y = np.asarray(Y, dtype=float)
    w = np.ones(len(D)) if w is None else np.asarray(w, dtype=float)
    Z = np.hstack([X, B])
    gamma = weighted_probit(D, Z, w)
    pZ = norm.cdf(gamma[0] + Z @ gamma[1:])
    sel = D == 1
    Y1, pZ1, X1, w1 = Y[sel], pZ[sel], X[sel], w[sel]
    X1c = np.column_stack([np.ones(len(Y1)), X1])
    vecrho = np.asarray(vecrho, dtype=float)
    obj_rho = np.zeros(len(vecrho))
    for j, rhoa in enumerate(vecrho):
        obj = 0.0
        for tau in vectau:
            G = rotated_tau(tau, pZ1, rhoa)
            beta = weighted_rq(X1c, Y1, G, w1)
            obj += np.sum(w1 * pZ1 * ((Y1 <= X1c @ beta) - G)) / np.sum(w1)
        obj_rho[j] = obj ** 2
    rho = vecrho[np.argmin(obj_rho)]
    taus = np.asarray(taus, dtype=float)
    beta = np.zeros((X1c.shape[1], len(taus)))
    for j, tau in enumerate(taus):
        beta[:, j] = weighted_rq(X1c, Y1, rotated_tau(tau, pZ1, rho), w1)
    return {'rho': float(rho), 'beta': beta, 'taus': taus, 'objective': obj_rho, 'gamma': gamma}


def selection_qr_bootstrap(D, Y, X, B, reps=200, seed=None, **kwargs):
    '''Weighted (multiplier) bootstrap of selection_qr_weighted() with Exp(1) weights.
    Returns the point estimate, the bootstrap draws of rho and beta, and standard errors.'''
    rng = np.random.default_rng(seed)
    est = selection_qr_weighted(D, Y, X, B, **kwargs)
    n = len(np.asarray(D))
    rhos = np.zeros(reps)
    betas = np.zeros((reps,) + est['beta'].shape)
    for r in range(reps):
        b = selection_qr_weighted(D, Y, X, B, w=rng.exponential(size=n), **kwargs)
        rhos[r], betas[r] = b['rho'], b['beta']
    return {'estimate': est, 'rho_draws': rhos, 'beta_draws': betas,
            'rho_se': rhos.std(ddof=1), 'beta_se': betas.std(axis=0, ddof=1)}


# =============================================================================
# Rearrangement
# =============================================================================

def rearrange(beta, x):
    '''Monotone (rearranged) conditional quantiles at covariate values x.

    beta : (p, J) quantile coefficients on an increasing tau grid (a column per tau;
           first row the intercept, as returned by selection_qr()).
    x    : (p,) or (m, p) covariate rows including the leading 1.
    Returns the (J,) or (m, J) quantile curves x'beta(tau_j), each sorted over j
    (Chernozhukov, Fernandez-Val and Galichon 2010).
    '''
    q = np.atleast_2d(np.asarray(x, dtype=float)) @ np.asarray(beta, dtype=float)
    q = np.sort(q, axis=1)
    return q[0] if np.ndim(x) == 1 else q
