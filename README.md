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
Koenker (2002). See Portnoy and Koenker (1997, *Statistical Science*) and
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
selection_qr.py     probit(), bivariate_normal_cdf(), gaussian_copula_cdf(),
                    rotated_tau(), selection_qr()     <- Code_QRS.m
tests/test_rq.py    15 tests (pytest)
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

Not ported: `ipqr.m` (Koenker and Park's interior point method for nonlinear
quantile regression), which relies on MATLAB functions removed long ago
(`flops`, `fmin`). It would be a good next step:
`scipy.optimize.minimize_scalar` replaces `fmin` for the line search.

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
