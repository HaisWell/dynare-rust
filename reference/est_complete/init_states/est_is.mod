// estimate_initial_states_endogenous_prior tests (estimation completion): the AD test model
// with two extra states, s = 2a + b/2 (exactly collinear: singular unconditional covariance of
// the states, so that set_init_state.m projects the initial states) and q.
@#ifndef MH
@#define MH = 0
@#endif
@#include "is_model.inc"
varobs y c;

estimated_params;
stderr ea, 0.012, inv_gamma_pdf, 0.01, inf;
stderr eb, 0.018, inv_gamma_pdf, 0.02, inf;
stderr c, 0.0025, inv_gamma_pdf, 0.002, inf;
alpha, 0.31, beta_pdf, 0.33, 0.05;
rho, 0.85, beta_pdf, 0.85, 0.1;
rhob, 0.75, beta_pdf, 0.8, 0.1;
sigc, 1.8, gamma_pdf, 2, 0.5;
end;
@#ifdef USECAL
// Dynare 7.2 (do_parameter_initialization.m): the initial states are block six of xparam1 and
// take their calibration (M_.endo_initial_state.values = 0) under use_calibration (7.1 crashed).
estimated_params_init(use_calibration);
alpha, 0.30;
end;
@#endif

estimation(datafile='is_data.csv', nobs=120, mode_compute=0, mh_replic=@{MH}, mh_nblocks=1,
           posterior_sampling_method='slice', estimate_initial_states_endogenous_prior,
           prior_trunc=1e-10, nograph, nodiagnostic);
