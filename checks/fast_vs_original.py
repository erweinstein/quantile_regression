'''checks/fast_vs_original.py -- does selection_qr_fast() return the same estimates as
selection_qr(), and how much faster is it?

Same design as checks/selection_bootstrap_coverage.py (true rho = -0.5), 30 data sets
at n = 2,000 and 10 at n = 8,000. For each: rho from selection_qr(), from
selection_qr_fast() with the full grid, and with the coarse-then-fine search
(coarse_step 5 and 3, three best coarse points); the largest difference in the 19 x 2
quantile coefficients; and the times. Output: checks/fast_vs_original.csv.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
'''
import os, sys, time, csv, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from selection_qr import selection_qr  # noqa: E402
from selection_qr_fast import selection_qr_fast  # noqa: E402
warnings.simplefilter('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fast_vs_original.csv')


def data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n); B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    return D, 1 + X + (0.5 + 0.3 * X) * u, X, B


rows = []
for n, ns in ((2000, 30), (8000, 10)):
    for s in range(ns):
        D, Y, X, B = data(n, 7000 + s)
        t = time.time(); a = selection_qr(D, Y, X, B); ta = time.time() - t
        t = time.time(); f = selection_qr_fast(D, Y, X, B); tf = time.time() - t
        t = time.time(); c5 = selection_qr_fast(D, Y, X, B, search='coarse', coarse_step=5); tc5 = time.time() - t
        t = time.time(); c3 = selection_qr_fast(D, Y, X, B, search='coarse', coarse_step=3); tc3 = time.time() - t
        rows.append({'n': n, 'seed': 7000 + s, 'rho_orig': a['rho'], 'rho_fast': f['rho'], 'rho_coarse5': c5['rho'],
                     'rho_coarse3': c3['rho'], 'beta_maxdiff_fast': float(np.max(np.abs(a['beta'] - f['beta']))),
                     't_orig': ta, 't_fast': tf, 't_coarse5': tc5, 't_coarse3': tc3})
        print(rows[-1], flush=True)
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
