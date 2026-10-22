// analytic_derivation tests (estimation completion). STAGE selects the variant:
//  1: Bayesian, kalman_algo=0 (multivariate), lik_init=1, stderr/corr shocks + ME on c
//  2: as 1 with kalman_algo=2 (univariate filter)
//  3: correlated measurement errors on y and c, kalman_algo=2 (expanded state, Zflag=1)
//  4: maximum likelihood (no priors), kalman_algo=0
//  5: as 1 with lik_init=2
//  6: as 1 without measurement error (analytic_derivation_mode -1/-2)
@#ifndef STAGE
@#define STAGE = 1
@#endif
@#ifndef MC
@#define MC = 0
@#endif
@#ifndef AD
@#define AD = 0
@#endif
@#ifndef HESS
@#define HESS = -1
@#endif
@#ifndef COVA
@#define COVA = 1
@#endif
@#ifndef PEN
@#define PEN = 0
@#endif
@#include "ad_model.inc"
varobs y c;

@#if STAGE == 4
estimated_params;
stderr ea, 0.012;
stderr eb, 0.018;
corr ea, eb, 0.1;
stderr c, 0.0025;
alpha, 0.31;
rho, 0.85;
rhob, 0.75;
sigc, 1.8;
end;
@#else
estimated_params;
stderr ea, 0.012, inv_gamma_pdf, 0.01, inf;
stderr eb, 0.018, inv_gamma_pdf, 0.02, inf;
corr ea, eb, 0.1, normal_pdf, 0, 0.3;
@#if STAGE == 3
stderr y, 0.0015, inv_gamma_pdf, 0.001, inf;
stderr c, 0.0025, inv_gamma_pdf, 0.002, inf;
corr y, c, 0.1, normal_pdf, 0, 0.3;
@#elseif STAGE != 6
stderr c, 0.0025, inv_gamma_pdf, 0.002, inf;
@#endif
alpha, 0.31, beta_pdf, 0.33, 0.05;
rho, 0.85, beta_pdf, 0.85, 0.1;
rhob, 0.75, beta_pdf, 0.8, 0.1;
sigc, 1.8, gamma_pdf, 2, 0.5;
end;
@#endif

@#if STAGE == 2 || STAGE == 3
@#define KALG = 2
@#else
@#define KALG = 0
@#endif
@#if STAGE == 5
@#define LIKI = 2
@#else
@#define LIKI = 1
@#endif

// MC: mode_compute; AD: analytic_derivation; HESS >= 0: optim=('Hessian', HESS) for newrat;
// COVA: cova_compute; PEN: use_penalized_objective_for_hessian
estimation(datafile='ad_data.csv', nobs=200, kalman_algo=@{KALG}, lik_init=@{LIKI}, mode_compute=@{MC}, cova_compute=@{COVA}, nograph
@#if AD == 1
, analytic_derivation
@#endif
@#if HESS >= 0
, optim=('Hessian', @{HESS})
@#endif
@#if PEN == 1
, use_penalized_objective_for_hessian
@#endif
);
