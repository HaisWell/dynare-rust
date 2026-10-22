// reference (reference/q2/dump_gsa_q2.m): the NK model of reference/gsa/nk_gsa.mod with
// irf_calibration / moment_calibration (map_calibration.m) and identification after an estimation.
// Small New Keynesian model for the global sensitivity analysis reference
// (dump_gsa.m). The prior ranges contain determinate, indeterminate
// (phi_pi < 1) and explosive (rho_g > 1) regions.
// STAGE: 0 simulate data; 1 stability map; 2 stability map, prior_range=0;
// 3 reduced form, threshold; 4 Morris screening; 5 RMSE; 6 RMSE lik_only;
// 7 reduced form mapping without threshold (needs SS-ANOVA-R in MATLAB).

@#ifndef CASE
@#define CASE = 1
@#endif

var y pi i g u;
varexo e_g e_u e_i;
parameters sigma kappa beta phi_pi phi_y rho_i rho_g rho_u;

sigma = 1;
kappa = 0.1;
beta = 0.99;
phi_pi = 1.5;
phi_y = 0.125;
rho_i = 0.7;
rho_g = 0.9;
rho_u = 0.5;

model(linear);
y = y(+1) - sigma*(i - pi(+1)) + g;
pi = beta*pi(+1) + kappa*y + u;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
g = rho_g*g(-1) + e_g;
u = rho_u*u(-1) + e_u;
end;

shocks;
var e_g; stderr 0.5;
var e_u; stderr 0.3;
var e_i; stderr 0.2;
end;

steady;

varobs y pi i;

estimated_params;
stderr e_g, inv_gamma_pdf, 0.5, inf;
stderr e_u, inv_gamma_pdf, 0.3, inf;
stderr e_i, inv_gamma_pdf, 0.2, inf;
sigma, gamma_pdf, 1, 0.3;
kappa, gamma_pdf, 0.1, 0.05;
phi_pi, uniform_pdf, , , 0.5, 3;
phi_y, normal_pdf, 0.125, 0.05;
rho_i, beta_pdf, 0.7, 0.1;
rho_g, uniform_pdf, , , 0.5, 1.05;
rho_u, beta_pdf, 0.5, 0.2;
end;


@#if CASE == 1
irf_calibration;
y(1:4), e_g, [0, 3];
pi(1), e_u, +;
pi(2), e_u, [0.1, 0.4];
i(1), e_i, -;
end;

moment_calibration;
y, y, [0.5, 3];
pi, y(1), -;
y, pi(-(1:2)), [0, 1];
end;

sensitivity(Nsam=256, nodisplay, graph_format=none);
@#endif

@#if CASE == 2
estimation(datafile='nk_q2_data.csv', mode_compute=4, mh_replic=0, nograph, nodisplay);
identification(parameter_set=posterior_mode, no_identification_strength, nograph);
@#endif

@#if CASE == 3
// identification on the GSA sample (gsa_sample_file=1: nk_q2/gsa/nk_q2_prior.mat)
sensitivity(Nsam=32, nodisplay, graph_format=none);
identification(gsa_sample_file=1, no_identification_strength, nograph);
@#endif

@#if CASE == 4
identification(no_identification_strength, nograph);
identification(load_ident_files=1, no_identification_strength, nograph);
@#endif

@#if CASE == 5
sensitivity(identification=1, nodisplay, graph_format=none);
@#endif

@#if CASE == 6
sensitivity(identification=1, morris=2, Nsam=16, nodisplay, graph_format=none);
@#endif

@#if CASE == 7
// glue: Dynare 7.1's preprocessor has no `glue` option; the MATLAB run sets options_gsa.glue
// itself (verbatim), dynare-rust reads sensitivity(glue=1)
@#ifdef MATLAB_GLUE
verbatim;
options_gsa = struct('datafile', 'nk_q2_data.csv', 'rmse', 1, 'glue', 1, 'Nsam', 128, 'nodisplay', true);
options_gsa.graph_format = {'none'};
gsa.run(M_,oo_,options_,bayestopt_,estim_params_,options_gsa);
end;
@#else
sensitivity(datafile='nk_q2_data.csv', rmse=1, glue=1, Nsam=128, nodisplay, graph_format=none);
@#endif
@#endif

@#if CASE == 8
// ppost: the posterior sample of an MCMC (dynare-rust test only; Dynare reads _param_irf files)
estimation(datafile='nk_q2_data.csv', mode_compute=4, mh_replic=2000, mh_nblocks=1, mh_jscale=0.5, nograph, nodisplay);
irf_calibration;
pi(1), e_u, +;
end;
sensitivity(ppost=1, pprior=0, nodisplay, graph_format=none);
@#endif
