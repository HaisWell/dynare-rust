// Maximum-likelihood test model for dynare-rust (estimated_params without priors),
// stationary (lik_init = 1) with a measurement error on x. Data: make_data.py.
var x z;
varexo ex ez;
parameters a b c;
a = 0.8; b = 0.3; c = 0.5;

model(linear);
x = a*x(-1) + b*z(-1) + ex;
z = c*z(-1) + ez;
end;

shocks;
var ex; stderr 1;
var ez; stderr 0.5;
end;

steady;

varobs x z;

estimated_params;
a, 0.7, -0.99, 0.99;
b, 0.2, -2, 2;
c, 0.4, -0.99, 0.99;
stderr ex, 0.8, 0.01, 10;
stderr ez, 0.6, 0.01, 10;
stderr x, 0.2, 0.001, 5;
end;

@#ifdef POSTEST
// prefilter + post-estimation objects (reference/opts/dump_postest.m). POSTEST = 1: forecast=H,
// shock_decomposition (also over the forecast), realtime_shock_decomposition,
// conditional_forecast at the calibration; 2-4: what Dynare 7.1 rejects under prefilter.
// ml_mode.mat: the mode of the default (no define) ML estimation.
@#if POSTEST == 1
estimation(datafile='model_ml_obs.csv', prefilter=1, mode_file='ml_mode', mode_compute=0, nograph, smoother, forecast=6, conf_sig=0.8) z x;
shock_decomposition(nograph);
shock_decomposition(forecast_type=unconditional, nograph) x z;
realtime_shock_decomposition(forecast=3, save_realtime=[150 200], nograph) x z;
conditional_forecast_paths;
var x;
periods 1:2, 3;
values 0.5, 0.4;
end;
conditional_forecast(parameter_set=calibration, controlled_varexo=(ex), periods=6, replic=50, conf_sig=0.8);
@#elseif POSTEST == 2
estimation(datafile='model_ml_obs.csv', prefilter=1, mode_file='ml_mode', mode_compute=0, nograph, smoother);
forecast(periods=5) x z;
@#elseif POSTEST == 3
estimation(datafile='model_ml_obs.csv', prefilter=1, mode_file='ml_mode', mode_compute=0, nograph, smoother);
conditional_forecast_paths;
var x;
periods 1:2;
values 0.5;
end;
conditional_forecast(parameter_set=mle_mode, controlled_varexo=(ex), periods=6, replic=50);
@#else
estimation(datafile='model_ml_obs.csv', prefilter=1, mode_file='ml_mode', mode_compute=0, nograph, smoother, forecast=4) x;
@#endif
@#else
@#ifdef PREFILTER
// options test (reference/opts/dump_reference.m): demeaned data + endogenous prior
estimation(datafile='model_ml_obs.csv', prefilter=1, endogenous_prior, mode_compute=0, nograph, smoother);
@#else
estimation(datafile='model_ml_obs.csv', mode_compute=4, nograph);
@#endif
@#endif
