# Positive level data for model_rbc.mod (rbc_levels.csv) and their logs (rbc_logs.csv):
# AR(1) log deviations around the calibrated steady state. Seeded; any data will do, the
# test compares likelihoods.
import math
import random

random.seed(11)
css, yss = 2.306618, 3.015325
u = v = 0.0
lev, logs = [], []
for t in range(160):
    u = 0.9 * u + 0.008 * random.gauss(0, 1)
    v = 0.85 * v + 0.6 * u + 0.006 * random.gauss(0, 1)
    if t >= 40:
        lc, ly = math.log(css) + u, math.log(yss) + v
        logs.append((lc, ly))
        lev.append((math.exp(lc), math.exp(ly)))
for name, rows in [('rbc_levels.csv', lev), ('rbc_logs.csv', logs)]:
    with open(name, 'w') as f:
        f.write('c,y\n')
        for a, b in rows:
            f.write(f'{a!r},{b!r}\n')
