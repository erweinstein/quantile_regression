# Quantile regression in Python: a port of `rq.m`, `rqic.m`, and Arellano–Bonhomme's selection-corrected QR

Eliot R. Weinstein  
eliotw@uchicago.edu  
[erweinstein.com](https://erweinstein.com)

Repository: https://github.com/erweinstein/quantile_regression

Started as my final project for MSBI 31600 (Summer 2020): porting the MATLAB
interior point algorithm for quantile regression, and Arellano and Bonhomme's
selection-corrected quantile regression built on it, to Python. Cleaned up and
completed in 2026.

**Credit.** The algorithm is the Frisch–Newton interior point method for
quantile regression of Roger Koenker's `quantreg` package: `lp_fnm` by Daniel
Morillo and Roger Koenker, translated from Ox to MATLAB by Paul Eilers (1999),
modified by Roger Koenker (2000–) and Paul Eilers (2004); `rqic.m` by Roger
Koenker (2002); `ipqr.m` implements Koenker and Park (1996). See Portnoy and Koenker (1997, *Statistical Science*) and
<http://www.econ.uiuc.edu/~roger/research/rq/rq.html>. Selection-corrected
quantile regression and `Code_QRS.m` are by Manuel Arellano and Stéphane
Bonhomme (Arellano and Bonhomme, 2017, *Econometrica* 85(1), 1–28).

## Files

```
LICENSE             BSD 3-Clause License
NOTICE.md           what the license covers, and attribution
rq.py               bound(), lp_fnm(), rq()           <- rq.m (rq_fnm, lp_fnm, bound)
                    lp_fnm_ineq(), rqic()             <- rqic.m (inequality constraints R b >= r)
                    rq_objective()                    the check-function objective
ipqr.py             ipqr(), qr_objective()            <- ipqr.m, ipqr_objfunc.m (nonlinear QR)
selection_qr.py     probit(), bivariate_normal_cdf(), gaussian_copula_cdf(),
                    rotated_tau(), selection_qr()     <- Code_QRS.m
tests/test_rq.py    19 tests (pytest)
```

## Usage

```python
import numpy as np
from rq import rq, rqic
from selection_qr import selection_qr

X = np.column_stack([np.ones(n), x1, x2])     # include the intercept column yourself, as in rq.m
b_median = rq(X, y, 0.5)
b_90 = rq(X, y, 0.9)
b_c = rqic(X, y, R=[[0, 1, 0]], r=[0], tau=0.5)   # median regression with slope on x1 >= 0

res = selection_qr(D, Y, X, B)   # D: participation, X: covariates, B: excluded covariates
res['rho'], res['beta']          # copula parameter; (1 + k) x 19 coefficients for tau = .05..95
```

Requires numpy and scipy (and pytest for the tests). Run the tests with
`python -m pytest tests`.

## Checks

- `rq()` reaches the same objective value as scipy's HiGHS linear programming
  solver on the primal problem, for tau = 0.1, …, 0.9. (Solutions need not be
  unique; see FAQ 1 in `Quantreg_FAQ.txt`: the interior point method lands near
  the centroid of the solution set, the simplex at a vertex.)
- `rq()` on a constant gives the sample median.
- `rqic()` equals `rq()` when the constraint is slack and respects it when
  binding.
- `bivariate_normal_cdf()` agrees with `scipy.stats.multivariate_normal.cdf`
  to 1e-12.
- `selection_qr()` recovers the copula parameter and the quantile
  coefficients in a simulated selection model.

## What changed from the 2020 version, and why

The 2020 code (`rq-python.txt` and its two backups) followed `rq.m` closely and
its comments explained each MATLAB-to-Python decision; that approach is kept.
These are the fixes needed to make it run and give the right answer:

1. **Imports.** `from numpy import *` together with `numpy.` and `linalg.`
   prefixes (and the typo `lingalg`) left names undefined. Now: `import numpy
   as np`, and everything is called through `np.` (wildcard imports also
   shadow Python's built-in `min`, `max`, and `sum`, which caused trouble
   below). `h5py` and `scipy.sparse` were not needed.
2. **`bound()`.** `.nonzero()` already returns 0-based indices, so the `f = f - 1`
   shift (meant to undo MATLAB's 1-based indexing) pointed at the wrong
   elements. A boolean mask replaces `find()` with no index arithmetic at all.
3. **Indicator expressions.** `r = r + 0.001 * (r == 0)` and
   `z = r .* (r > 0)` are elementwise in MATLAB. `if r == 0:` compares a whole
   array (an error in numpy), so these became `0.001 * (r == 0)` and
   `np.where(r > 0, r, 0)`.
4. **Step-length minimums.** `min[fx, fs]` (square brackets) is a syntax
   error; MATLAB's `min(fx, fs)` is elementwise and `min(min(beta * fp), 1)`
   then takes the overall minimum. The helper `_step_lengths()` does exactly
   that.
5. **Operators.** `@` was used for scalar-times-vector products (`fd @ dz`)
   and `mu @ ...`; those are plain `*`. `^` is bitwise XOR in Python; MATLAB's
   `^3` is `**3`.
6. **Control flow.** The step `x = x + fp * dx` etc. sat in an `else:` branch,
   so it was skipped whenever the step was modified, and `return y` was inside
   the `while` loop, so the solver stopped after one iteration. In MATLAB the
   step is taken on every iteration, after the `if` block.
7. **Shapes.** Transposes of 1-D numpy arrays do nothing, so `y.T`, `r.T`,
   etc. were silently ignored. All vectors are now 1-D and the MATLAB
   transposes are dropped (the comments show the original lines). `A' \ c'`
   with a non-square `A'` is least squares (`np.linalg.lstsq`), not `solve`.
8. **The diagonal matrix Q.** `spdiags(sqrt(q), 0, n, n)` only rescales, so
   `A * Q` is `A * sqrt(q)` column by column and `Q * r'` is `sqrt(q) * r`; no
   sparse matrix (and no transpose) needed.
9. **`rq()` inputs.** `numpy.ones(m, 1)` is not the numpy signature
   (`np.ones(m)`), and `numpy.asarray(X)` without assignment does nothing. The
   `assert`s for all-zero inputs are replaced by `ValueError` checks for
   things that actually break the algorithm: tau outside (0, 1), a
   rank-deficient design, mismatched lengths. (Asserts are skipped when
   Python runs with `-O`.)
10. **Selection QR (`SSQR.txt` → `selection_qr.py`).**
    - `hstack(X, B)` needs one tuple: `np.hstack([X, B])`.
    - `numpy.arrange(0.20, 0.80, 0.20)` is `arange`, and it excludes the
      endpoint, so 0.8 was missing.
    - The probit (`glmfit`) is a few lines of Fisher scoring, so no
      statsmodels is needed.
    - MATLAB's `copulacdf('Gaussian', ...)` is the bivariate normal cdf at
      normal quantiles, computed exactly and fast with Owen's T function
      (`scipy.special.owens_t`), so no copula package is needed.
    - The rotated quantile levels are kept inside (0, 1), which the interior
      point method requires.
    - The grid search and the final estimates follow `Code_QRS.m` (squared
      objective; `SSQR.txt` used the absolute value, which has the same
      minimizer).

`ipqr.m` (Koenker and Park's interior point method for nonlinear quantile
regression) is ported in `ipqr.py`. The MATLAB version passed the model as a
function *name* and found its Jacobian by prepending `d` to it; in Python the
two functions are passed directly. MATLAB's long-removed `flops` is dropped,
and `fmin` (the line search) becomes `scipy.optimize.minimize_scalar` on the
same interval. With a linear model, `ipqr()` reaches the same objective as
`rq()`; the tests also check a nonlinear exponential model.

## Extensions (beyond the replication)

The files above replicate `rq.m`, `rqic.m`, `ipqr.m` and `Code_QRS.m` and are
left as they are. The files below add what the original code does not have:
standard errors and bootstrap inference, a faster computation of the same
estimates, the paper's own search settings, monotone quantile curves, and a
smoothed estimator for large data. Each comes with a check in `checks/`; the
numbers below are from those scripts. They ran on a 2-core cloud machine with
other jobs running, so the absolute times are rough and the ratios are what matter.

```
qr_extensions.py      rq_se()                  standard errors for rq(): Powell's kernel sandwich
                                               (as quantreg's summary.rq(se = "ker")) and an iid version
                      weighted_rq(), weighted_probit(), selection_qr_weighted()
                      selection_qr_bootstrap() weighted (multiplier) bootstrap for selection-corrected QR
                      rearrange()              monotone quantile curves (Chernozhukov et al. 2010)
selection_qr_fast.py  rq_fast()                rq() with the same iterations, vectorized
                      rq_preprocessed()        Portnoy-Koenker preprocessing for observation-specific tau
                      selection_qr_fast()      selection_qr(), faster; settings = "paper"; coarse search
smoothed_qr.py        conquer(), conquer_bootstrap()   convolution-smoothed QR with multiplier bootstrap
checks/               the simulations quoted below
tests/                test_extensions.py, test_fast_and_smoothed.py (29 tests in all pass)
```

**Standard errors for `rq()`** (`checks/conquer_coverage.py`; n = 1,000, two
covariates, t₄ errors whose scale grows with |x₁|, 300 data sets). Coverage of
nominal 95% intervals for the three coefficients at the median: kernel sandwich
96.7-97.3% (a little conservative), iid formula 86.0% for the slope whose
spread depends on x₁ (93.7-94.3% for the others). At τ = 0.9 the kernel
sandwich covered 87.7-93.0% and the iid formula 93.3-95.0%. Neither is right
everywhere, which is the usual experience with these formulas; the bootstrap
below did better.

**Smoothed quantile regression** (`smoothed_qr.py`; He, Pan, Tan and Zhou 2023).
It agrees with the R package `conquer` (version 1.3.3) to about 4 × 10⁻⁴ on the
same data, the package's own convergence tolerance. In the same 300 data sets, its multiplier
bootstrap (300 draws, Exp(1) weights) gave standard errors within 5% of the
estimates' Monte Carlo SD and 95% intervals covering 92.0-95.3% (normal),
91.7-94.7% (percentile) and 90.7-93.7% (pivotal); the estimates' Monte Carlo
SDs were 1-5% smaller than `rq()`'s. It is a different estimator (it depends on
the bandwidth h, by default the package's max{((log n + p)/n)^0.4, 0.05}), so it
is offered beside `rq()`, not instead of it.

**Faster selection-corrected QR** (`selection_qr_fast.py`, `checks/timing.py`,
`checks/fast_vs_original.py`). Following Pereda-Fernández (2024): the
interior point iterations vectorized; fits along the copula grid started from
the neighbouring copula value, with Portnoy and Koenker's (1997) preprocessing
adapted to observation-specific quantile levels; and an optional coarse-then-fine
search over the copula grid. Median times, 3 data sets per size:

| n | `selection_qr()` | fast, exact (default) | with preprocessing | preprocessing + coarse search |
|---|---|---|---|---|
| 2,000 | 4.7 s | 4.2 s (1.1×) | 3.1 s (1.5×) | 1.4 s (3.3×) |
| 8,000 | 17.9 s | 14.3 s (1.2×) | 5.6 s (3.2×) | 2.9 s (6.2×) |
| 32,000 | 81.8 s | 60.5 s (1.4×) | 13.7 s (6.9×) | 13.1 s (6.8×) |

The default (no preprocessing, full grid) returns `selection_qr()`'s estimates
to rounding error (the same ρ in all 22 data sets checked; the coefficients to
about 10⁻¹⁵). With preprocessing every rotated regression reaches the same
optimum, but ρ moved by one or two grid steps (0.02-0.04) in 16 of 30 data sets
of 2,000 and 1 of 16 of 8,000 or more. The reason is that p observations lie
exactly on each fitted line, and whether `1{Y ≤ X'b}` counts them depends on the
last digits the solver leaves; `Code_QRS.m` does not settle it either (counting
them as on or below the line, `tie_tol = 1e-7`, moved ρ by 0.02-0.04 in 9 of 12
data sets of 2,000). These shifts are small next to ρ's sampling spread (a
bootstrap SE of about 0.11 at n = 1,000), but they are not the same number, so preprocessing is an option,
recommended for large samples and for bootstrapping. The speed-ups are smaller
than the 45-125× Pereda-Fernández reports for his Algorithm 2, which also
uses a reduced quantile grid for the slopes; ours keep `Code_QRS.m`'s grids.

**The paper's settings** (`selection_qr_fast(..., settings = "paper")`): the
copula search over the nine deciles and 200 grid points, as Arellano and
Bonhomme describe for their application, instead of `Code_QRS.m`'s four
quantiles (0.2-0.8) and 99 points. With preprocessing and the coarse search it
took 4.5-15.7 s at n = 2,000-32,000, and ρ came out within 0.005 of the code
settings' value in 7 of the 9 data sets timed (0.015 in the other two).

**Rearrangement** (`rearrange()`, `checks/rearrangement.py`; 100 data sets each at
n = 500 and 2,000). Where the linear quantile model is right (x = 0 and 2), the
19 estimated quantiles crossed in 0-1% of data sets. At x = -2 the true quantile
curve is decreasing in the model's form, so the estimated curves always cross;
sorting them cut the root mean squared error from 0.115 to 0.076 (n = 500) and
from 0.103 to 0.069 (n = 2,000). In line with Chernozhukov, Fernández-Val and
Galichon's result, it was never worse in any data set.

**Bootstrap for selection-corrected QR** (`selection_qr_bootstrap()`,
`checks/selection_bootstrap_coverage.py`): BOOTSTRAP_RESULT

## Data note

Arellano and Bonhomme's applications use, among others, the data of Huber and
Melly (2015), "A Test of the Conditional Independence Assumption in Sample
Selection Models," *Journal of Applied Econometrics* 30(7), available from the
JAE data archive: <http://qed.econ.queensu.ca/jae/2015-v30.7/huber-melly/>.

## References

- Arellano, M. and S. Bonhomme (2017), "Quantile Selection Models With an
  Application to Understanding Changes in Wage Inequality," *Econometrica*
  85(1), 1–28.
- Koenker, R. (2005), *Quantile Regression*, Cambridge University Press.
- Koenker, R. and B. J. Park (1996), "An Interior Point Algorithm for
  Nonlinear Quantile Regression," *Journal of Econometrics* 71, 265–283.
- Chernozhukov, V., I. Fernández-Val and A. Galichon (2010), "Quantile and
  Probability Curves Without Crossing," *Econometrica* 78(3), 1093–1125.
- He, X., X. Pan, K. M. Tan and W.-X. Zhou (2023), "Smoothed Quantile
  Regression with Large-Scale Inference," *Journal of Econometrics* 232(2),
  367–388.
- Ma, S. and M. R. Kosorok (2005), "Robust Semiparametric M-Estimation and the
  Weighted Bootstrap," *Journal of Multivariate Analysis* 96(1), 190–217.
- Pereda-Fernández, S. (2024), "Fast Algorithms for Quantile Regression with
  Selection," arXiv:2402.16693.
- Powell, J. L. (1991), "Estimation of Monotonic Regression Models under
  Quantile Restrictions," in Barnett, Powell and Tauchen (eds.),
  *Nonparametric and Semiparametric Methods in Econometrics and Statistics*,
  Cambridge University Press, 357–384.
- Portnoy, S. and R. Koenker (1997), "The Gaussian Hare and the Laplacian
  Tortoise: Computability of Squared-Error versus Absolute-Error Estimators,"
  *Statistical Science* 12(4), 279–300.

## License

Copyright (c) 2020, 2026 Eliot R. Weinstein. The Python code is licensed under
the BSD 3-Clause License (`LICENSE`), the license of NumPy, SciPy, pandas,
scikit-learn, and statsmodels. It covers this port only: the algorithms and the
original MATLAB code remain the work of Koenker, Morillo, and Eilers (rq.m,
rqic.m) and of Arellano and Bonhomme (Code_QRS.m), subject to their own terms;
see `NOTICE.md`.
