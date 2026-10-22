// Estimation stage switch (see ESTIMATION CALL at the end); no -D = 3, the MCMC.
@#ifndef STAGE
@#define STAGE = 3
@#endif
// =========================================================================
//  model_gm05.mod  --  small open economy New Keynesian model
// -------------------------------------------------------------------------
//  Small open economy reference model of dynare-rust (see
//  docs/REFERENCES.md). The model of
//
//    Gali, J. and T. Monacelli (2005), "Monetary Policy and Exchange Rate
//    Volatility in a Small Open Economy", Review of Economic Studies 72(3),
//
//  in the estimable form of
//
//    Lubik, T. and F. Schorfheide (2007), "Do Central Banks Respond to
//    Exchange Rate Movements? A Structural Investigation", Journal of
//    Monetary Economics 54(4), 1069-1087 (equations 1-6):
//
//  an open-economy IS curve and Phillips curve with terms-of-trade terms, a
//  policy rule on inflation, output and nominal depreciation, and AR(1)
//  terms of trade, world output, world inflation and technology growth.
//  Purely forward-looking, no lags beyond one: the structural contrast to
//  the QPM (model_qpm.mod) and SW07 (sw07.mod) replacements. The level of
//  the nominal exchange rate is observed (a unit root => diffuse_filter).
//  Units: quarterly log deviations in the model; observables in percent
//  (annualised for inflation and the interest rate).
//  The data (model_gm05_obs.csv) are SIMULATED from this model at its
//  calibration (gen_data.m, fixed seed); no real-world data.
// =========================================================================

var y pi r de dq le ystar pistar z ybar
    dy_obs pi_obs r_obs le_obs dq_obs;

varexo e_r e_q e_z e_ys e_ps;

parameters psi1 psi2 psi3 rho_r alpha r_a kappa tau rho_q rho_z rho_ys rho_ps pi_a gamma_q;

psi1   = 1.30;
psi2   = 0.23;
psi3   = 0.14;
rho_r  = 0.69;
alpha  = 0.11;
r_a    = 2.50;
kappa  = 0.32;
tau    = 0.31;
rho_q  = 0.31;
rho_z  = 0.42;
rho_ys = 0.97;
rho_ps = 0.40;
pi_a   = 3.00;
gamma_q = 0.50;


model;
# beta = exp(-r_a/400);
# tau_a = tau + alpha*(2 - alpha)*(1 - tau);

// open-economy IS curve
y = y(+1) - tau_a*(r - pi(+1)) - rho_z*z - alpha*tau_a*dq(+1)
    + alpha*(2 - alpha)*((1 - tau)/tau)*(ystar(+1) - ystar);
// open-economy Phillips curve
pi = beta*pi(+1) + alpha*beta*dq(+1) - alpha*dq + (kappa/tau_a)*(y - ybar);
// potential output under flexible prices
ybar = -alpha*(2 - alpha)*((1 - tau)/tau)*ystar;
// policy rule
r = rho_r*r(-1) + (1 - rho_r)*(psi1*pi + psi2*y + psi3*de) + e_r;
// nominal depreciation from CPI inflation, the terms of trade and world inflation
de = pi - (1 - alpha)*dq - pistar;
le = le(-1) + de;

dq = rho_q*dq(-1) + e_q;
ystar = rho_ys*ystar(-1) + e_ys;
pistar = rho_ps*pistar(-1) + e_ps;
z = rho_z*z(-1) + e_z;

// measurement
dy_obs = gamma_q + y - y(-1) + z;
pi_obs = pi_a + 4*pi;
r_obs = pi_a + r_a + 4*r;
le_obs = le;
dq_obs = dq;
end;


// =========================================================================
// STEADY STATE  (le is a unit root: level 0 chosen)
// =========================================================================
initval;
dy_obs = gamma_q;
pi_obs = pi_a;
r_obs  = pi_a + r_a;
end;

steady(solve_algo=1, maxit=1000);


// =========================================================================
// OBSERVED VARIABLES
// =========================================================================
varobs dy_obs pi_obs r_obs le_obs dq_obs;


// =========================================================================
// SHOCKS  (calibration used to simulate model_gm05_obs.csv)
// =========================================================================
shocks;
var e_r;  stderr 0.30;
var e_q;  stderr 1.50;
var e_z;  stderr 0.60;
var e_ys; stderr 1.00;
var e_ps; stderr 1.50;
end;


// =========================================================================
// ESTIMATED PARAMETERS  (priors in the spirit of Lubik-Schorfheide 2007, Table 1)
// =========================================================================
estimated_params;
psi1,        1.30, gamma_pdf,     1.50, 0.50;
psi2,        0.23, gamma_pdf,     0.25, 0.13;
psi3,        0.14, gamma_pdf,     0.25, 0.13;
rho_r,       0.69, beta_pdf,      0.50, 0.20;
alpha,       0.11, beta_pdf,      0.20, 0.05;
r_a,         2.50, gamma_pdf,     2.50, 1.00;
kappa,       0.32, gamma_pdf,     0.50, 0.25;
tau,         0.31, beta_pdf,      0.50, 0.20;
rho_q,       0.31, beta_pdf,      0.40, 0.20;
rho_z,       0.42, beta_pdf,      0.20, 0.10;
rho_ys,      0.97, beta_pdf,      0.90, 0.05;
rho_ps,      0.40, beta_pdf,      0.80, 0.10;
pi_a,        3.00, gamma_pdf,     3.00, 1.00;
gamma_q,     0.50, normal_pdf,    0.50, 0.25;
stderr e_r,  0.30, inv_gamma_pdf, 0.50, 0.50;
stderr e_q,  1.50, inv_gamma_pdf, 1.50, 1.50;
stderr e_z,  0.60, inv_gamma_pdf, 1.00, 1.00;
stderr e_ys, 1.00, inv_gamma_pdf, 1.50, 1.50;
stderr e_ps, 1.50, inv_gamma_pdf, 0.55, 0.55;
end;

@#if STAGE == 1
estimated_params_init(use_calibration);
end;
@#endif


// =========================================================================
// ESTIMATION CALL
// =========================================================================
// STAGED RUN. `dynare model_gm05 -DSTAGE=<n>`; no -D = stage 3.
//   1: mode_compute=4 (csminwel) from the calibration, mh_replic=0.
//   2: mode_compute=6 from the stage-1 mode (model_gm05_mode_stage1.mat), mode_check.
//   3: MCMC from the stage-2 mode.
@#if STAGE == 1
estimation(datafile='model_gm05_obs.csv', mode_compute=4, mh_replic=0,
           graph_format=(pdf), nodisplay, first_obs=1, diffuse_filter)
     y;
@#elseif STAGE == 2
estimation(datafile='model_gm05_obs.csv', mode_compute=6, mh_replic=0,
           mode_file='model_gm05/Output/model_gm05_mode_stage1',
           plot_priors=1, mode_check, graph_format=(pdf), nodisplay,
           first_obs=1, diffuse_filter)
     y;
@#else
estimation(datafile='model_gm05_obs.csv', mode_compute=0,
           mode_file='model_gm05/Output/model_gm05_mode',
           graph_format=(pdf,eps), nodisplay,
           mh_replic=2000000, mh_nblocks=6, filtered_vars,
           filter_step_ahead=[1:12], forecast=8, conf_sig=0.70,
           first_obs=1, diffuse_filter, smoothed_state_uncertainty)
     y;
@#endif

@#if STAGE == 3
generate_trace_plots([1:6]);
@#endif
