// Small nonlinear RBC model for dynare-rust's tests of estimation options:
// STAGE 1: loglinear (data in levels, logged by makedataset);
// STAGE 2: loglinear + logdata (data already in logs);
// STAGE 3: loglinear + endogenous_prior;
// STAGE 4: loglinear, post-estimation objects: forecast=H, shock_decomposition (also over the
//          forecast), conditional_forecast at the mode (reference/opts/dump_postest.m);
// STAGE 5: loglinear, the forecast command and conditional_forecast(parameter_set=calibration);
// STAGE 4 and 5 start from rbc_mode.mat, the mode found by STAGE 6.
@#ifndef STAGE
@#define STAGE = 1
@#endif
var c k a y;
varexo e;
parameters alpha beta delta rho;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;

model;
1/c = beta/c(+1)*(alpha*a(+1)*k^(alpha-1) + 1 - delta);
c + k = y + (1-delta)*k(-1);
y = a*k(-1)^alpha;
log(a) = rho*log(a(-1)) + e;
end;

initval;
a = 1; k = 30; c = 2.3; y = 3;
end;
steady;

shocks;
var e; stderr 0.01;
var y; stderr 0.005;
end;

varobs c y;

estimated_params;
rho, 0.85, beta_pdf, 0.8, 0.1;
alpha, 0.3, normal_pdf, 0.33, 0.05;
stderr e, 0.012, inv_gamma_pdf, 0.01, inf;
stderr y, 0.004, inv_gamma_pdf, 0.01, inf;
end;

@#if STAGE == 1
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, smoother);
@#elseif STAGE == 2
estimation(datafile='rbc_logs.csv', loglinear, logdata, mode_compute=0, mh_replic=0, nograph, smoother);
@#elseif STAGE == 3
estimation(datafile='rbc_levels.csv', loglinear, endogenous_prior, mode_compute=0, mh_replic=0, nograph, smoother);
@#elseif STAGE == 4
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=0, nograph, smoother, forecast=6, conf_sig=0.8) y c k a;
shock_decomposition(nograph) y c k;
shock_decomposition(parameter_set=calibration, forecast_type=unconditional, nograph) y c;
conditional_forecast_paths;
var y;
periods 1:2, 3;
values 1.12, 1.13;
end;
conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(e), periods=6, replic=50, conf_sig=0.8);
@#elseif STAGE == 5
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=0, nograph, smoother);
forecast(periods=5, conf_sig=0.8) y c a;
conditional_forecast_paths;
var y;
periods 1:2, 3;
values 1.12, 1.13;
end;
conditional_forecast(parameter_set=calibration, controlled_varexo=(e), periods=6, replic=50, conf_sig=0.8);
@#else
// STAGE 6: the mode behind rbc_mode.mat (dump_postest.m copies model_rbc/Output/model_rbc_mode.mat)
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=4, mh_replic=0, nograph);
@#endif
