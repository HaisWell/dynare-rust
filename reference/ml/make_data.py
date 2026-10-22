# Simulates model_ml.mod's data (seeded): x_t = 0.8 x_{t-1} + 0.3 z_{t-1} + ex_t,
# z_t = 0.5 z_{t-1} + ez_t, sd(ex) = 1, sd(ez) = 0.5; x observed with a measurement
# error of sd 0.3. 50 burn-in periods, 200 kept, a few missing values.
import random

random.seed(7)
x = z = 0.0
rows = []
for t in range(250):
    z_prev = z
    z = 0.5 * z_prev + 0.5 * random.gauss(0, 1)
    x = 0.8 * x + 0.3 * z_prev + random.gauss(0, 1)
    if t >= 50:
        rows.append((x + 0.3 * random.gauss(0, 1), z))
with open('model_ml_obs.csv', 'w') as f:
    f.write('x,z\n')
    for k, (a, b) in enumerate(rows):
        xs = '' if k in (20, 21, 120) else repr(a)
        f.write(f'{xs},{b!r}\n')
