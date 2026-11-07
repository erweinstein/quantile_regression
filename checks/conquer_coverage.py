'''checks/conquer_coverage.py -- coverage of conquer_bootstrap()'s 95% intervals, and of
rq() with rq_se() (kernel and iid) for comparison.

Design: n = 1,000; y = 1 + x1 - 0.5 x2 + (1 + 0.5|x1|) e, x ~ N(0, I_2), e ~ t_4 (so the
quantile slopes on x1 vary with tau: the truth is 1 + 0.5 q_tau sign-adjusted, computed
by simulation). tau = 0.5 and 0.9; 300 data sets; 300 bootstrap draws each.
The true conditional quantile coefficients are found from one sample of 2,000,000.
Output: checks/conquer_coverage.csv.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
'''
import os, sys, csv, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from smoothed_qr import conquer_bootstrap, conquer  # noqa: E402
from qr_extensions import rq_se  # noqa: E402
from rq import rq  # noqa: E402
warnings.simplefilter('ignore')
N, R, REPS = 1000, 300, 300


def draw(n, rng):
    x = rng.normal(size=(n, 2))
    return np.column_stack([np.ones(n), x]), 1 + x @ [1, -0.5] + (1 + 0.5 * np.abs(x[:, 0])) * rng.standard_t(4, size=n)


rng = np.random.default_rng(99)
Xb, yb = draw(2_000_000, rng)
rows = []
for tau in (0.5, 0.9):
    truth = conquer(Xb, yb, tau, h=0.01)            # population quantile regression (h small, n huge)
    for d in range(R):
        rng = np.random.default_rng(10_000 * int(10 * tau) + d)
        X, y = draw(N, rng)
        bs = conquer_bootstrap(X, y, tau, reps=REPS, seed=d)
        b = rq(X, y, tau)
        row = {'tau': tau, 'dataset': d}
        for k in range(3):
            for kind in ('percentile', 'pivotal', 'normal'):
                lo, hi = bs[kind][k]
                row[f'b{k}_{kind}'] = int(lo <= truth[k] <= hi)
            row[f'b{k}_conquer_est'] = bs['estimate'][k]
            row[f'b{k}_conquer_se'] = bs['se'][k]
            row[f'b{k}_rq_est'] = b[k]
        for m in ('ker', 'iid'):
            se, _ = rq_se(X, y, tau, b, m)
            for k in range(3):
                row[f'b{k}_rq_{m}'] = int(abs(b[k] - truth[k]) <= 1.96 * se[k])
                row[f'b{k}_rq_{m}_se'] = se[k]
        row.update({f'truth{k}': truth[k] for k in range(3)})
        rows.append(row)
        if d % 25 == 0:
            print(tau, d, flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conquer_coverage.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
