// Estimation options of the estimation-completion stream (reference/est_complete/options/dump_options.m):
// STAGE 1: dr=cycle_reduction; 2: dr=logarithmic_reduction; 3: qz_zero_threshold;
// STAGE 4: no measurement error (2 observables, 1 shock): initial_estimation_checks error;
// STAGE 5: no measurement error, y observed in the first period only: error in first period;
// STAGE 6: STAGE 5 + no_init_estimation_check_first_obs (warning, the estimation runs);
// STAGE 7: shock calibrated with a zero variance: initial_estimation_checks error;
// STAGE 8: frequentist_smoother at the mode of rbc_mode.mat (no `smoother` option).
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
@#if STAGE == 7
var e; stderr 0;
@#else
var e; stderr 0.01;
@#endif
@#if STAGE < 4 || STAGE > 6
var y; stderr 0.005;
@#endif
end;

varobs c y;

estimated_params;
rho, 0.85, beta_pdf, 0.8, 0.1;
alpha, 0.3, normal_pdf, 0.33, 0.05;
@#if STAGE != 7
stderr e, 0.012, inv_gamma_pdf, 0.01, inf;
@#endif
@#if STAGE < 4 || STAGE > 6
stderr y, 0.004, inv_gamma_pdf, 0.01, inf;
@#endif
end;

@#if STAGE == 1
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, dr=cycle_reduction, dr_cycle_reduction_tol=1e-10);
@#elseif STAGE == 2
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, dr=logarithmic_reduction, dr_logarithmic_reduction_maxiter=200);
@#elseif STAGE == 3
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, qz_zero_threshold=1e-8);
@#elseif STAGE == 4
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph);
@#elseif STAGE == 5
estimation(datafile='rbc_firstobs.csv', loglinear, mode_compute=0, mh_replic=0, nograph);
@#elseif STAGE == 6
estimation(datafile='rbc_firstobs.csv', loglinear, mode_compute=0, mh_replic=0, nograph, no_init_estimation_check_first_obs);
@#elseif STAGE == 7
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph);
@#elseif STAGE == 8
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=0, nograph, frequentist_smoother);
// dynare-rust only (tests/est_complete_options.rs):
@#elseif STAGE == 9
// MCMC with moments_varendo, contemporaneous_correlation, smoother; posterior_nograph
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=400, mh_nblocks=1,
           mh_jscale=0.5, moments_varendo, contemporaneous_correlation, ar=1, smoother, sub_draws=100,
           posterior_nograph, plot_priors=0, nodiagnostic, graph_format=svg) c y;
@#elseif STAGE == 10
// as 9, no_posterior_kernel_density, posterior_graph after posterior_nograph
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=400, mh_nblocks=1,
           mh_jscale=0.5, moments_varendo, smoother, sub_draws=100, no_posterior_kernel_density,
           posterior_nograph, posterior_graph, plot_priors=0, nodiagnostic, graph_format=svg) c y;
@#elseif STAGE == 11
// frequentist_smoother = false: no classical smoother
estimation(datafile='rbc_levels.csv', loglinear, mode_file='rbc_mode', mode_compute=0, mh_replic=0, nograph, smoother, frequentist_smoother=false);
@#elseif STAGE == 12
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, estimate_initial_states_endogenous_prior);
@#elseif STAGE == 13
estimation(datafile='rbc_levels.csv', loglinear, mode_compute=0, mh_replic=0, nograph, estimate_initial_states_endogenous_prior,
           posterior_sampling_method='slice');
@#endif
