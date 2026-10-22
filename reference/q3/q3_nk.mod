// (estimation options) reference model: the 3-equation NK model of reference/mcmc.
//   -DSTAGE=1  DSGE-VAR, calibrated weight (dsge_var = 3, dsge_varlag = 2)
//   -DSTAGE=2  DSGE-VAR, estimated dsge_prior_weight
//   -DSTAGE=3  DSGE-VAR, dsge_var = Inf
//   -DSTAGE=4  heteroskedastic_filter (stationary init, multivariate filter), smoother
//   -DSTAGE=5  heteroskedastic_filter, diffuse_filter + univariate (kalman_algo = 4), smoother
//   -DSTAGE=6  rescale_prediction_error_covariance
//   -DSTAGE=7  conditional_likelihood
//   -DSTAGE=8  lyapunov = fixed_point (lik_init = 1)
//   -DSTAGE=9  lyapunov = doubling
//   -DSTAGE=10 DSGE-VAR with estimated weight: mode, MCMC, bayesian_irf (statistical comparison)
//   -DSTAGE=12 filter_step_ahead = [1 3], filter_covariance, filter_decomposition (multivariate smoother)
//   -DSTAGE=13 the same with diffuse_filter, kalman_algo = 4 (univariate smoother)
//   -DSTAGE=14 filter_initial_state (stationary init), smoother
//   -DSTAGE=15 filter_initial_state, diffuse_filter, smoother
//   -DSTAGE=16 skewed shocks (calibrated skew e_y, estimated skew e_pi), kalman_algo = 5
//   -DSTAGE=17 the same with skewed_kalman_prune_tol = 0 (no pruning) and rescale_prediction_error_covariance
//   -DSTAGE=18 generalized priors (shifted beta/gamma/weibull/inverse gamma, truncated normal bounds)
@#include "nk_model.inc"
varobs y pi i;

@#ifndef STAGE
@#define STAGE = 1
@#endif

@#if STAGE == 18
estimated_params;
rho_y, 0.6, beta_pdf, 0.6, 0.15, 0.1, 0.95;
kappa, 0.15, gamma_pdf, 0.15, 0.05, 0.02;
phi_pi, 1.4, 1.1, 2.5, normal_pdf, 1.5, 0.25, 1.05, 3;
rho_i, 0.5, weibull_pdf, 0.5, 0.2, 0.05;
stderr e_y, 0.8, inv_gamma_pdf, 0.8, inf, 0.1;
stderr e_pi, 0.5, inv_gamma2_pdf, 0.5, 0.4, 0.05;
stderr e_i, 0.3, uniform_pdf, , , 0, 2;
end;
@#else
estimated_params;
rho_y, beta_pdf, 0.6, 0.15;
kappa, gamma_pdf, 0.15, 0.05;
phi_pi, 1.4, 1, 3, normal_pdf, 1.5, 0.25;
rho_i, beta_pdf, 0.5, 0.2;
stderr e_y, inv_gamma_pdf, 0.8, inf;
stderr e_pi, inv_gamma_pdf, 0.5, inf;
stderr e_i, uniform_pdf, , , 0, 2;
@#if STAGE == 2 || STAGE == 10
dsge_prior_weight, 1.5, uniform_pdf, , , 0.5, 4;
@#endif
@#if STAGE == 16 || STAGE == 17
skew e_pi, -0.3, normal_pdf, 0, 0.3;
@#endif
end;
@#endif

@#if STAGE == 4 || STAGE == 5
heteroskedastic_shocks;
var e_y;
periods 50:60, 61;
scales 2, 0.5;
var e_pi;
periods 100, 101:103;
values 0.1, 1.2;
var e_i;
periods 120:125;
scales 0.5;
end;
@#endif

@#if STAGE == 16 || STAGE == 17
shocks;
skew e_y = 0.6;
end;
@#endif

@#if STAGE == 14 || STAGE == 15
filter_initial_state;
y(0) = 0.5;
y(-1) = 0.2;
i(0) = -0.3;
end;
@#endif

@#if STAGE == 1
estimation(datafile = nk_data, first_obs = 5, dsge_var = 3, dsge_varlag = 2, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 2
estimation(datafile = nk_data, first_obs = 5, dsge_var, dsge_varlag = 2, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 3
estimation(datafile = nk_data, first_obs = 3, dsge_var = Inf, dsge_varlag = 1, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 4
estimation(datafile = nk_data, heteroskedastic_filter, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 5
estimation(datafile = nk_data, first_obs = 2, heteroskedastic_filter, diffuse_filter, kalman_algo = 4, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 6
estimation(datafile = nk_data, rescale_prediction_error_covariance, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 7
estimation(datafile = nk_data, conditional_likelihood, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 8
estimation(datafile = nk_data, lyapunov = fixed_point, lyapunov_fixed_point_tol = 1e-12, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 9
estimation(datafile = nk_data, lyapunov = doubling, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 10
estimation(datafile = nk_data, first_obs = 5, dsge_var, dsge_varlag = 2, mode_compute = 4, mh_replic = 6000, mh_nblocks = 2,
           mh_jscale = 0.5, bayesian_irf, irf = 8, nograph, nodiagnostic);
@#elseif STAGE == 12
estimation(datafile = nk_data, mode_compute = 0, mh_replic = 0, smoother, filtered_vars, filter_step_ahead = [1 3],
           filter_covariance, filter_decomposition, nograph);
@#elseif STAGE == 13
estimation(datafile = nk_data, mode_compute = 0, mh_replic = 0, smoother, filtered_vars, filter_step_ahead = [1 3],
           filter_covariance, filter_decomposition, diffuse_filter, kalman_algo = 4, nograph);
@#elseif STAGE == 14
estimation(datafile = nk_data, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 15
estimation(datafile = nk_data, mode_compute = 0, mh_replic = 0, smoother, diffuse_filter, nograph);
@#elseif STAGE == 16
estimation(datafile = nk_data, nobs = 60, kalman_algo = 5, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 17
estimation(datafile = nk_data, nobs = 60, kalman_algo = 5, skewed_kalman_prune_tol = 0, rescale_prediction_error_covariance,
           mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 18
estimation(datafile = nk_data, mode_compute = 0, mh_replic = 0, nograph);
@#endif
