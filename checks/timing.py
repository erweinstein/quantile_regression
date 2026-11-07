'''checks/timing.py -- wall-clock times of selection_qr() and selection_qr_fast() (with
and without preprocessing; full and coarse copula search; Code_QRS.m's and the paper's
search settings), 3 data sets per size, same design as the other checks.
Output: checks/timing.csv. Times depend on the machine and on other jobs running.

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


def data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n); B = rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    return D, 1 + X + (0.5 + 0.3 * X) * u, X, B


RUNS = {'original': lambda d: selection_qr(*d),
        'fast_exact': lambda d: selection_qr_fast(*d, preprocess=False),
        'fast_preprocessed': lambda d: selection_qr_fast(*d, preprocess=True),
        'fast_preprocessed_coarse': lambda d: selection_qr_fast(*d, preprocess=True, search='coarse'),
        'paper_settings_preprocessed': lambda d: selection_qr_fast(*d, settings='paper', preprocess=True),
        'paper_settings_preprocessed_coarse': lambda d: selection_qr_fast(*d, settings='paper', preprocess=True, search='coarse')}
rows = []
for n in (2000, 8000, 32000):
    for s in range(3):
        d = data(n, 9000 + s)
        for name, f in RUNS.items():
            t = time.time(); r = f(d); el = time.time() - t
            rows.append({'n': n, 'seed': 9000 + s, 'run': name, 'seconds': round(el, 2), 'rho': r['rho']})
            print(rows[-1], flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'timing.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
