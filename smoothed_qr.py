'''smoothed_qr.py -- convolution-smoothed quantile regression ("conquer"; Fernandes,
Guerre and Horta 2021; He, Pan, Tan and Zhou 2023) with a multiplier bootstrap.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com

The check function rho_tau(u) has a kink at 0, which is why quantile regression is a
linear program (rq.py). Smoothing it by convolution with a kernel K_h,
l_h(u) = (rho_tau * K_h)(u), gives a convex, twice-differentiable loss whose
derivative, with the Gaussian kernel, is l_h'(u) = tau - Phi(-u / h). The estimator

    beta_h = argmin_b (1/n) sum_i l_h(y_i - x_i'b)

is then found by Newton's method (p is small here; the conquer package uses
gradient descent with Barzilai-Borwein steps, which scales to large p). He et al.
show that with h of order ((p + log n) / n)^(2/5) its bias is negligible, it is
asymptotically equivalent to the quantile regression estimator, and it can be more
efficient in finite samples; and that a multiplier bootstrap (refitting with
random weights of mean 1 and variance 1) gives valid intervals, computed quickly
because each refit is again a smooth convex problem. This is a different estimator
from rq() (it depends on h), so it is offered beside rq(), not instead of it.

Default bandwidth as in the conquer R package: h = max(((log n + p) / n)^0.4, 0.05).

References
- Fernandes M, Guerre E, Horta E (2021). Smoothing quantile regressions. Journal of
  Business & Economic Statistics 39(1), 338-357.
- He X, Pan X, Tan KM, Zhou W-X (2023). Smoothed quantile regression with large-scale
  inference. Journal of Econometrics 232(2), 367-388.
- Tan KM, Wang L, Zhou W-X (2022). High-dimensional quantile regression: convolution
  smoothing and concave regularization. JRSS B 84(1), 205-233.
'''

import numpy as np
from scipy.stats import norm


def default_bandwidth(n, p):
    return max(((np.log(n) + p) / n) ** 0.4, 0.05)


def _loss(r, tau, h):
    # Gaussian-smoothed check loss: E rho_tau(r + h Z) = h phi(r/h) + r (tau - Phi(-r/h))
    return np.mean(h * norm.pdf(r / h) + r * (tau - norm.cdf(-r / h)))


def conquer(X, y, tau=0.5, h=None, w=None, b0=None, tol=1e-8, max_iter=100):
    '''Smoothed quantile regression (Gaussian kernel) by damped Newton steps.

    X : (n, p) design (include the intercept column yourself, as for rq()).
    w : optional nonnegative weights (the multiplier bootstrap uses these).
    Returns the coefficient vector.'''
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    h = default_bandwidth(n, p) if h is None else h
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0] if b0 is None else np.asarray(b0, dtype=float)

    def wloss(bb):
        r = y - X @ bb
        return np.mean(w * (h * norm.pdf(r / h) + r * (tau - norm.cdf(-r / h))))

    f = wloss(b)
    for _ in range(max_iter):
        r = y - X @ b
        g = X.T @ (w * (norm.cdf(-r / h) - tau)) / n
        H = (X * (w * norm.pdf(r / h) / h)[:, None]).T @ X / n
        try:
            step = np.linalg.solve(H + 1e-12 * np.eye(p), g)
        except np.linalg.LinAlgError:
            step = g
        t = 1.0
        while True:                                   # backtracking line search (the loss is convex)
            bn = b - t * step
            fn = wloss(bn)
            if fn <= f - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t /= 2
        b, f = bn, fn
        if np.max(np.abs(t * step)) < tol:
            break
    return b


def conquer_bootstrap(X, y, tau=0.5, h=None, reps=1000, alpha=0.05, seed=None):
    '''Multiplier bootstrap: refit with Exp(1) weights (mean 1, variance 1), each refit
    started from the full-sample estimate. Returns the estimate, the draws, standard
    errors, and percentile, pivotal and normal intervals (as in conquer's
    ci = "bootstrap").'''
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    h = default_bandwidth(n, p) if h is None else h
    b = conquer(X, y, tau, h)
    draws = np.array([conquer(X, y, tau, h, w=rng.exponential(size=n), b0=b) for _ in range(reps)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    se = draws.std(axis=0, ddof=1)
    z = norm.ppf(1 - alpha / 2)
    return {'estimate': b, 'draws': draws, 'se': se, 'h': h,
            'percentile': np.column_stack([lo, hi]),
            'pivotal': np.column_stack([2 * b - hi, 2 * b - lo]),
            'normal': np.column_stack([b - z * se, b + z * se])}
