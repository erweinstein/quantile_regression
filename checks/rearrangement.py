'''checks/rearrangement.py -- how often do selection_qr()'s estimated quantile curves
cross, and how much does rearrangement (Chernozhukov, Fernandez-Val and Galichon 2010)
change the error?

Same design as the other checks (true quantile curve at x: 1 + x + (0.5 + 0.3 x) z_tau),
n = 500 and 2,000, 100 data sets each (the fast estimator without preprocessing, which
reproduces selection_qr()). For x = -2, 0 and 2: the share of data sets with any
crossing over tau = 0.05, ..., 0.95, and the root mean squared error of the curve
before and after rearrangement. Output: checks/rearrangement.csv.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
'''
import os, sys, csv, warnings
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from selection_qr_fast import selection_qr_fast  # noqa: E402
from qr_extensions import rearrange  # noqa: E402
warnings.simplefilter('ignore')
rows = []
for n in (500, 2000):
    for s in range(100):
        rng = np.random.default_rng(5000 + s)
        X = rng.normal(size=n); B = rng.normal(size=n)
        u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
        D = (0.3 + 0.5 * X + B + v > 0).astype(int)
        Y = 1 + X + (0.5 + 0.3 * X) * u
        r = selection_qr_fast(D, Y, X, B, preprocess=False)
        z = norm.ppf(r['taus'])
        for x in (-2.0, 0.0, 2.0):
            truth = 1 + x + (0.5 + 0.3 * x) * z
            if 0.5 + 0.3 * x < 0:
                truth = np.sort(truth)
            q = np.array([1.0, x]) @ r['beta']
            qr = rearrange(r['beta'], np.array([1.0, x]))
            rows.append({'n': n, 'seed': 5000 + s, 'x': x, 'crossing': int(np.any(np.diff(q) < 0)),
                         'rmse_raw': float(np.sqrt(np.mean((q - truth) ** 2))),
                         'rmse_rearranged': float(np.sqrt(np.mean((qr - truth) ** 2)))})
    print(n, flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rearrangement.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
