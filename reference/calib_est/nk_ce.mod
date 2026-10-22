// irf_calibration / moment_calibration inside the estimation (dsge_likelihood.m's
// endogenous_prior_restrictions check, info 49), the prior sampler and the GSA stability
// mapping. Reference: reference/calib_est/dump_calib_est.m (ref_calib_est.json, logs).
// The NK model of reference/q2/nk_q2.mod (same data, nk_ce_data.csv).
// CASE 1: estimation, mode + MCMC, restrictions that bind in the MCMC (interior mode);
// CASE 2: a moment restriction that binds at the unrestricted mode (simplex; mode on the boundary);
// CASE 3: prior simulate (Endogenous prior violation share);
// CASE 4: prior moments with a restriction violated at the prior mode;
// CASE 5: sensitivity: stability mapping with some draws matching the restrictions.

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

@#if CASE == 1 || CASE == 3
irf_calibration;
y(1), e_g, [0, 1.6];
pi(1:2), e_u, [0.1, 0.45];
i(1), e_i, +;
end;

moment_calibration;
y, pi, [0, 0.75];
y, y(-1), [0.5, 0.95];
end;
@#endif

@#if CASE == 1
estimation(datafile='nk_ce_data.csv', mode_compute=4, mh_replic=2000, mh_nblocks=2, mh_jscale=0.5, nograph, nodisplay);
@#endif

@#if CASE == 2
moment_calibration;
y, pi, [0, 0.69];
end;
// Dynare's simplex: deterministic (csminwel's cliff perturbations draw random numbers; newrat's
// gradient step differs on this model, restrictions or not)
estimation(datafile='nk_ce_data.csv', mode_compute=8, mh_replic=0, nograph, nodisplay);
@#endif

@#if CASE == 3
prior simulate;
@#endif

@#if CASE == 4
irf_calibration;
y(1), e_g, [0, 0.5];
end;
prior moments;
@#endif

@#if CASE == 5
irf_calibration;
i(1), e_i, +;
pi(1), e_u, +;
end;

moment_calibration;
y, y(-1), [0.5, 0.95];
end;

sensitivity(Nsam=256, nodisplay, graph_format=none);
@#endif
