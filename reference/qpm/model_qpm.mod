// Estimation stage switch (see ESTIMATION CALL at the end); no -D = 3, the MCMC.
@#ifndef STAGE
@#define STAGE = 3
@#endif
// =========================================================================
//  model_qpm.mod  --  quarterly projection model (QPM) of a small open
//                     economy, with a foreign block
// -------------------------------------------------------------------------
//  Semi-structural reference model of dynare-rust (see
//  docs/REFERENCES.md). The gap model of
//
//    Berg, A., P. Karam and D. Laxton (2006), "A Practical Model-Based
//    Approach to Monetary Policy Analysis -- Overview" (IMF WP/06/80) and
//    "Practical Model-Based Monetary Policy Analysis -- A How-To Guide"
//    (IMF WP/06/81):
//
//  an IS curve, a Phillips curve on four-quarter inflation (lead 4, lags 1-4),
//  a forward-looking policy rule on expected four-quarter inflation (lead 3),
//  uncovered interest parity with partly backward-looking exchange-rate
//  expectations, and stochastic trends for potential output and the real
//  exchange rate (two unit roots => diffuse_filter). The foreign economy is a
//  smaller closed version of the same model. Units: annualised percent for
//  growth, inflation and interest rates; 100*log for levels.
//  The data (model_qpm_obs.csv) are SIMULATED from this model at its
//  calibration (gen_data.m, fixed seed); no real-world data.
// =========================================================================

var
    // home: gaps and trends
    ygap l_gdp_bar g_bar dl_gdp
    pie pie4 pie_tar pie4_exp
    rs rr rr_bar rr_gap
    l_z l_z_bar dl_z_bar z_gap z_e prem dl_s
    // foreign
    ygap_us pie_us pie4_us rs_us rr_us rr_bar_us rr_gap_us;

varexo e_ygap e_pie e_rs e_z e_l_z_bar e_dl_z_bar e_rr_bar e_l_gdp_bar e_g e_tar e_exp
       e_ygap_us e_pie_us e_rs_us e_rr_bar_us;

parameters
    b_lag b_ld b_rr b_z b_us
    a_ld a_y a_z
    g_rs g_pie g_y
    d_z
    rho_rr rho_dz rho_tar tau_g
    bu_lag bu_ld bu_rr au_ld au_y gu_rs gu_pie gu_y rho_rr_us
    g_ss pie_tar_ss rr_bar_ss pie_tar_us rr_bar_us_ss;

b_lag  = 0.70;  b_ld  = 0.15;  b_rr  = 0.15;  b_z   = 0.05;  b_us  = 0.20;
a_ld   = 0.40;  a_y   = 0.25;  a_z   = 0.10;
g_rs   = 0.70;  g_pie = 1.50;  g_y   = 0.40;
d_z    = 0.50;
rho_rr = 0.90;  rho_dz = 0.80; rho_tar = 0.90; tau_g = 0.10;
bu_lag = 0.80;  bu_ld = 0.10;  bu_rr = 0.10;
au_ld  = 0.30;  au_y  = 0.15;
gu_rs  = 0.80;  gu_pie = 1.50; gu_y  = 0.50;
rho_rr_us = 0.90;
g_ss = 2.5;  pie_tar_ss = 3.0;  rr_bar_ss = 1.5;  pie_tar_us = 2.0;  rr_bar_us_ss = 0.5;


model;
// =========================================================================
// HOME ECONOMY
// =========================================================================
// IS curve: lagged real interest rate gap and real exchange rate gap, foreign demand
ygap = b_lag*ygap(-1) + b_ld*ygap(+1) - b_rr*rr_gap(-1) + b_z*z_gap(-1) + b_us*ygap_us + e_ygap;

// Phillips curve on four-quarter inflation; real depreciation passes through
pie = a_ld*pie4(+4) + (1 - a_ld)*pie4(-1) + a_y*ygap(-1) + a_z*(z_gap - z_gap(-1)) + e_pie;
pie4 = (pie + pie(-1) + pie(-2) + pie(-3))/4;
pie_tar = rho_tar*pie_tar(-1) + (1 - rho_tar)*pie_tar_ss + e_tar;
// one-year-ahead inflation expectations (survey), with a measurement shock
pie4_exp = pie4(+4) + e_exp;

// policy rule on expected four-quarter inflation three quarters ahead
rs = g_rs*rs(-1) + (1 - g_rs)*(rr_bar + pie4(+3) + g_pie*(pie4(+3) - pie_tar(+3)) + g_y*ygap) + e_rs;
rr = rs - pie(+1);
rr_gap = rr - rr_bar;
rr_bar = rho_rr*rr_bar(-1) + (1 - rho_rr)*rr_bar_ss + e_rr_bar;

// UIP with partly backward-looking expectations; risk premium from the trend differential
z_e = d_z*l_z(+1) + (1 - d_z)*(l_z(-1) + 2*dl_z_bar/4);
l_z = z_e - (rr - rr_us - prem)/4 + e_z;
prem = rr_bar - rr_bar_us - dl_z_bar;
z_gap = l_z - l_z_bar;
l_z_bar = l_z_bar(-1) + dl_z_bar/4 + e_l_z_bar;
dl_z_bar = rho_dz*dl_z_bar(-1) + e_dl_z_bar;
// nominal depreciation, annualised
dl_s = 4*(l_z - l_z(-1)) + pie - pie_us;

// potential output: level (deviation from the drift) and growth
l_gdp_bar = l_gdp_bar(-1) + (g_bar - g_ss)/4 + e_l_gdp_bar;
g_bar = tau_g*g_ss + (1 - tau_g)*g_bar(-1) + e_g;
dl_gdp = g_ss + 4*(l_gdp_bar - l_gdp_bar(-1) + ygap - ygap(-1));

// =========================================================================
// FOREIGN ECONOMY
// =========================================================================
ygap_us = bu_lag*ygap_us(-1) + bu_ld*ygap_us(+1) - bu_rr*rr_gap_us(-1) + e_ygap_us;
pie_us = au_ld*pie4_us(+4) + (1 - au_ld)*pie4_us(-1) + au_y*ygap_us(-1) + e_pie_us;
pie4_us = (pie_us + pie_us(-1) + pie_us(-2) + pie_us(-3))/4;
rs_us = gu_rs*rs_us(-1) + (1 - gu_rs)*(rr_bar_us + pie4_us(+3) + gu_pie*(pie4_us(+3) - pie_tar_us) + gu_y*ygap_us) + e_rs_us;
rr_us = rs_us - pie_us(+1);
rr_gap_us = rr_us - rr_bar_us;
rr_bar_us = rho_rr_us*rr_bar_us(-1) + (1 - rho_rr_us)*rr_bar_us_ss + e_rr_bar_us;
end;


// =========================================================================
// STEADY STATE  (l_gdp_bar and l_z_bar are unit roots: level 0 chosen)
// =========================================================================
initval;
dl_gdp   = g_ss;
g_bar    = g_ss;
pie      = pie_tar_ss;
pie4     = pie_tar_ss;
pie_tar  = pie_tar_ss;
pie4_exp = pie_tar_ss;
rr_bar   = rr_bar_ss;
rr       = rr_bar_ss;
rs       = rr_bar_ss + pie_tar_ss;
prem     = rr_bar_ss - rr_bar_us_ss;
dl_s     = pie_tar_ss - pie_tar_us;
pie_us   = pie_tar_us;
pie4_us  = pie_tar_us;
rr_bar_us = rr_bar_us_ss;
rr_us    = rr_bar_us_ss;
rs_us    = rr_bar_us_ss + pie_tar_us;
end;

steady(solve_algo=1, maxit=1000);


// =========================================================================
// OBSERVED VARIABLES
// =========================================================================
varobs
    dl_gdp
    pie
    rs
    l_z
    pie4_exp
    ygap_us
    pie_us
    rs_us;


// =========================================================================
// SHOCKS  (calibration used to simulate model_qpm_obs.csv)
// =========================================================================
shocks;
var e_ygap;       stderr 0.50;
var e_pie;        stderr 1.00;
var e_rs;         stderr 0.40;
var e_z;          stderr 2.00;
var e_l_z_bar;    stderr 0.50;
var e_dl_z_bar;   stderr 0.30;
var e_rr_bar;     stderr 0.20;
var e_l_gdp_bar;  stderr 0.20;
var e_g;          stderr 0.30;
var e_tar;        stderr 0.10;
var e_exp;        stderr 0.30;
var e_ygap_us;    stderr 0.40;
var e_pie_us;     stderr 0.60;
var e_rs_us;      stderr 0.30;
var e_rr_bar_us;  stderr 0.15;
end;


// =========================================================================
// ESTIMATED PARAMETERS
// =========================================================================
estimated_params;
b_lag,          0.70,  beta_pdf,      0.70, 0.10;
b_ld,           0.15,  beta_pdf,      0.15, 0.05;
b_rr,           0.15,  gamma_pdf,     0.15, 0.05;
b_z,            0.05,  gamma_pdf,     0.05, 0.02;
b_us,           0.20,  gamma_pdf,     0.20, 0.10;
a_ld,           0.40,  beta_pdf,      0.40, 0.10;
a_y,            0.25,  gamma_pdf,     0.25, 0.10;
a_z,            0.10,  gamma_pdf,     0.10, 0.05;
g_rs,           0.70,  beta_pdf,      0.70, 0.10;
g_pie,          1.50,  gamma_pdf,     1.50, 0.30;
g_y,            0.40,  gamma_pdf,     0.40, 0.15;
d_z,            0.50,  beta_pdf,      0.50, 0.15;
rho_dz,         0.80,  beta_pdf,      0.80, 0.10;
bu_lag,         0.80,  beta_pdf,      0.75, 0.10;
bu_rr,          0.10,  gamma_pdf,     0.10, 0.05;
au_ld,          0.30,  beta_pdf,      0.30, 0.10;
au_y,           0.15,  gamma_pdf,     0.15, 0.05;
gu_rs,          0.80,  beta_pdf,      0.75, 0.10;
gu_pie,         1.50,  gamma_pdf,     1.50, 0.30;
gu_y,           0.50,  gamma_pdf,     0.50, 0.15;
stderr e_ygap,       0.50, inv_gamma_pdf, 0.50, 0.50;
stderr e_pie,        1.00, inv_gamma_pdf, 1.00, 1.00;
stderr e_rs,         0.40, inv_gamma_pdf, 0.40, 0.40;
stderr e_z,          2.00, inv_gamma_pdf, 2.00, 2.00;
stderr e_l_z_bar,    0.50, inv_gamma_pdf, 0.50, 0.50;
stderr e_dl_z_bar,   0.30, inv_gamma_pdf, 0.30, 0.30;
stderr e_rr_bar,     0.20, inv_gamma_pdf, 0.20, 0.20;
stderr e_l_gdp_bar,  0.20, inv_gamma_pdf, 0.20, 0.20;
stderr e_g,          0.30, inv_gamma_pdf, 0.30, 0.30;
stderr e_tar,        0.10, inv_gamma_pdf, 0.10, 0.10;
stderr e_exp,        0.30, inv_gamma_pdf, 0.30, 0.30;
stderr e_ygap_us,    0.40, inv_gamma_pdf, 0.40, 0.40;
stderr e_pie_us,     0.60, inv_gamma_pdf, 0.60, 0.60;
stderr e_rs_us,      0.30, inv_gamma_pdf, 0.30, 0.30;
stderr e_rr_bar_us,  0.15, inv_gamma_pdf, 0.15, 0.15;
end;

@#if STAGE == 1
estimated_params_init(use_calibration);
end;
@#endif


// =========================================================================
// ESTIMATION CALL
// =========================================================================
// STAGED RUN. `dynare model_qpm -DSTAGE=<n>`; no -D = stage 3.
//   1: mode_compute=4 (csminwel) from the calibration, mh_replic=0.
//   2: mode_compute=6 from the stage-1 mode (model_qpm_mode_stage1.mat), mode_check.
//   3: MCMC from the stage-2 mode.
@#if STAGE == 1
estimation(datafile='model_qpm_obs.csv', mode_compute=4, mh_replic=0,
           graph_format=(pdf), nodisplay, first_obs=1, diffuse_filter)
     ygap;
@#elseif STAGE == 2
estimation(datafile='model_qpm_obs.csv', mode_compute=6, mh_replic=0,
           mode_file='model_qpm/Output/model_qpm_mode_stage1',
           plot_priors=1, mode_check, graph_format=(pdf), nodisplay,
           first_obs=1, diffuse_filter)
     ygap;
@#else
estimation(datafile='model_qpm_obs.csv', mode_compute=0,
           mode_file='model_qpm/Output/model_qpm_mode',
           graph_format=(pdf,eps), nodisplay,
           mh_replic=2000000, mh_nblocks=6, filtered_vars,
           filter_step_ahead=[1:12], forecast=8, conf_sig=0.70,
           first_obs=1, diffuse_filter, smoothed_state_uncertainty)
     ygap;
@#endif

@#if STAGE == 3
generate_trace_plots([1:6]);
@#endif
