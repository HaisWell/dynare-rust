// Stream X test model: lik_init 2/4/5, observation_trends, .m datafiles,
// estimated_params_bounds / estimated_params_remove. Data: make_data.py.
// -DLIKINIT=k (default 1), -DTREND (observation_trends on x, slope g estimated),
// -DPREFILTER, -DUNITROOT (w a random walk, observed), -DMFILE (x_lik_data.m instead of the csv).
@#ifndef LIKINIT
@#define LIKINIT = 1
@#endif
var x z w;
varexo ex ez ew;
parameters a b c g d;
a = 0.8; b = 0.3; c = 0.5; g = 0.02; d = 0.1;

model(linear);
x = a*x(-1) + b*z(-1) + ex;
z = c*z(-1) + ez;
@#ifdef UNITROOT
w = w(-1) + d*x + ew;
@#else
w = 0.6*w(-1) + d*x + ew;
@#endif
end;

shocks;
var ex; stderr 1;
var ez; stderr 0.5;
var ew; stderr 0.1;
end;

steady;

@#ifdef UNITROOT
varobs x z w;
@#else
varobs x z;
@#endif

@#ifdef TREND
observation_trends;
x (g);
end;
@#endif

estimated_params;
a, 0.7, -0.99, 0.99;
b, 0.2, -2, 2;
c, 0.4, -0.99, 0.99;
d, 0.1, -5, 5;
stderr ex, 0.8, 0.01, 10;
stderr ez, 0.6, 0.01, 10;
stderr ew, 0.1, 0.001, 10;
@#ifdef TREND
g, 0.015, -1, 1;
@#endif
end;

// stream X: the bounds of a and c are changed, d is not estimated after all
estimated_params_bounds;
a, -0.95, 0.95;
stderr ez, 0.02, 8;
end;
estimated_params_remove;
d;
end;

@#ifdef MFILE
@#define DATAFILE = "x_lik_data.m"
@#else
@#define DATAFILE = "x_lik_obs.csv"
@#endif

@#ifdef PREFILTER
estimation(datafile='@{DATAFILE}', lik_init=@{LIKINIT}, prefilter=1, mode_compute=0, nograph, smoother);
@#else
estimation(datafile='@{DATAFILE}', lik_init=@{LIKINIT}, mode_compute=0, nograph, smoother);
@#endif
