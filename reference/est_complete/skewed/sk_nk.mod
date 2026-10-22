// Estimation completion (skewed Kalman filter, kalman_algo = 5) reference model: the NK model
// of reference/q3 with skewed shocks (calibrated skew e_y, estimated skew e_pi).
//   -DSTAGE=1  kalman_algo = 5, smoother at the initial values (kalman_smoother_pruned_skewed.m)
//   -DSTAGE=2  skewed_kalman_rank_deficiency_transform (csn_statespace_linear_transform.m)
//   -DSTAGE=3  skewed_kalman_mvnlogcdf = 'mvncdf', skewed_kalman_prune_tol = 0.1
//   -DSTAGE=4  missing observations (Gaussian missing_observations_kalman_filter, as Dynare)
//   -DSTAGE=5  lik_init = 2, smoother
//   -DSTAGE=6  lik_init = 4
//   -DSTAGE=7  lik_init = 5
//   -DSTAGE=8  measurement error on y, filter_initial_state, smoother
//   -DSTAGE=9  lik_init = 3 (error)
//   -DSTAGE=10 rank_deficiency_transform with mvncdf and lik_init = 2
@#include "nk_model.inc"
varobs y pi i;

@#ifndef STAGE
@#define STAGE = 1
@#endif

estimated_params;
rho_y, beta_pdf, 0.6, 0.15;
kappa, gamma_pdf, 0.15, 0.05;
phi_pi, 1.4, 1, 3, normal_pdf, 1.5, 0.25;
rho_i, beta_pdf, 0.5, 0.2;
stderr e_y, inv_gamma_pdf, 0.8, inf;
stderr e_pi, inv_gamma_pdf, 0.5, inf;
stderr e_i, uniform_pdf, , , 0, 2;
skew e_pi, -0.3, normal_pdf, 0, 0.3;
@#if STAGE == 8
stderr y, inv_gamma_pdf, 0.2, inf;
@#endif
end;

shocks;
skew e_y = 0.6;
end;

@#if STAGE == 8
filter_initial_state;
y(0) = 0.5;
y(-1) = 0.2;
i(0) = -0.3;
end;
@#endif

@#if STAGE == 1
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 2
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, skewed_kalman_rank_deficiency_transform, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 3
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, skewed_kalman_mvnlogcdf = 'mvncdf', skewed_kalman_prune_tol = 0.1, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 4
estimation(datafile = nk_data_missing, nobs = 40, kalman_algo = 5, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 5
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, lik_init = 2, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 6
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, lik_init = 4, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 7
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, lik_init = 5, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 8
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, mode_compute = 0, mh_replic = 0, smoother, nograph);
@#elseif STAGE == 9
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, diffuse_filter, mode_compute = 0, mh_replic = 0, nograph);
@#elseif STAGE == 10
estimation(datafile = nk_data, nobs = 40, kalman_algo = 5, lik_init = 2, skewed_kalman_rank_deficiency_transform, skewed_kalman_mvnlogcdf = 'mvncdf', skewed_kalman_prune_tol = 0.1, mode_compute = 0, mh_replic = 0, nograph);
@#endif
