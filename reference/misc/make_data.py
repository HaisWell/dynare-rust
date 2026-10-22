# Data of x_lik.mod (seeded): x_t = 0.8 x_{t-1} + 0.3 z_{t-1} + ex_t, z_t = 0.5 z_{t-1} + ez_t,
# w_t = w_{t-1} + 0.1 x_t (a random walk); x carries a linear trend 0.02*t (observation_trends
# test). 50 burn-in periods, 120 kept, two missing values of x.
# Also writes x_lik_data.m, the same series as a .m datafile.
import random

random.seed(11)
x = z = w = 0.0
rows = []
for t in range(170):
    z_prev = z
    z = 0.5 * z_prev + 0.5 * random.gauss(0, 1)
    x = 0.8 * x + 0.3 * z_prev + random.gauss(0, 1)
    w = w + 0.1 * x
    if t >= 50:
        k = t - 50
        rows.append((x + 0.02 * (k + 1), z, w))
with open('x_lik_obs.csv', 'w') as f:
    f.write('x,z,w\n')
    for k, (a, b, c) in enumerate(rows):
        xs = '' if k in (30, 31) else repr(a)
        f.write(f'{xs},{b!r},{c!r}\n')
with open('x_lik_data.m', 'w') as f:
    f.write('% x_lik.mod data as a .m datafile (make_data.py)\n')
    for j, name in enumerate(['x', 'z', 'w']):
        f.write(f'{name} = [\n')
        for k, r in enumerate(rows):
            v = 'NaN' if (j == 0 and k in (30, 31)) else repr(r[j])
            f.write(f'  {v}\n')
        f.write("]';\n")
    f.write('x = x(:);\n')
