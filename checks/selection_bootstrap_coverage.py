'''checks/selection_bootstrap_coverage.py -- does the weighted bootstrap of
selection_qr_weighted() give intervals with their nominal coverage?

Design: n = 1000; X, B ~ N(0, 1); (u, v) bivariate normal with correlation 0.5;
participation D = 1{0.3 + 0.5 X + B + v > 0} (about 60% participate); outcome
Y = 1 + X + (0.5 + 0.3 X) u. In Arellano and Bonhomme's parameterization the copula
parameter is rho = corr(u, -v) = -0.5 and beta(tau) = (1 + 0.5 z_tau, 1 + 0.3 z_tau).
50 data sets, 100 bootstrap draws each (Exp(1) weights). Reports the coverage of
95% normal (estimate +- 1.96 SE) and percentile intervals for rho and for beta at
tau = 0.25, 0.5, 0.75, and the ratio of the mean bootstrap SE to the Monte Carlo SD.
Results are appended to checks/selection_bootstrap_coverage.csv as each data set
finishes, so the run can be stopped and restarted.

With the argument "uniform", X ~ Uniform(-1, 1) instead of N(0, 1), so the spread
0.5 + 0.3 X stays between 0.2 and 0.8 and the linear quantile model is exactly right
(with N(0, 1), the spread is negative for about 5% of X). Output then goes to
checks/selection_bootstrap_coverage_uniform.csv.

Eliot R. Weinstein
eliotw@uchicago.edu
erweinstein.com
'''
import os, sys, csv
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qr_extensions import selection_qr_bootstrap  # noqa: E402

N, DATASETS, REPS = 1000, 50, 100
UNIFORM = len(sys.argv) > 1 and sys.argv[1] == 'uniform'
TAUS = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)
J = {0.25: 4, 0.5: 9, 0.75: 14}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'selection_bootstrap_coverage_uniform.csv' if UNIFORM else 'selection_bootstrap_coverage.csv')
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {int(r['dataset']) for r in csv.DictReader(f)}
for d in range(DATASETS):
    if d in done:
        continue
    rng = np.random.default_rng(1000 + d)
    X = rng.uniform(-1, 1, size=N) if UNIFORM else rng.normal(size=N)
    B = rng.normal(size=N)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=N).T
    D = (0.3 + 0.5 * X + B + v > 0).astype(int)
    Y = 1 + X + (0.5 + 0.3 * X) * u
    res = selection_qr_bootstrap(D, Y, X, B, reps=REPS, seed=2000 + d)
    est = res['estimate']
    row = {'dataset': d, 'rho_hat': est['rho'], 'rho_se': res['rho_se'],
           'rho_lo': np.quantile(res['rho_draws'], 0.025), 'rho_hi': np.quantile(res['rho_draws'], 0.975)}
    for tau, j in J.items():
        for k, name in enumerate(['b0', 'b1']):
            row[f'{name}_{tau}_hat'] = est['beta'][k, j]
            row[f'{name}_{tau}_se'] = res['beta_se'][k, j]
            row[f'{name}_{tau}_lo'] = np.quantile(res['beta_draws'][:, k, j], 0.025)
            row[f'{name}_{tau}_hi'] = np.quantile(res['beta_draws'][:, k, j], 0.975)
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print('data set', d, 'rho', est['rho'], 'se', round(res['rho_se'], 3), flush=True)
