// (reference/q2/dump_gsa_q2.m): reference/identification/nk_ident.mod with checks_via_subsets
// Small New Keynesian model for the identification tests of dynare-rust.
// The Phillips-curve slope is the product alpha*psi: the pair (alpha, psi) is
// not identified (only the product enters the model), the rest is.
// The gross rate rr = exp(r)/beta makes the model nonlinear and gives a
// parameter-dependent steady state (tests dYss and the g2 term of dg1).
@#ifndef CASE
@#define CASE = 1
@#endif

var y pi r g u rr;
varexo eg eu er;
parameters beta sigma alpha psi phipi phiy rhog rhou;

beta  = 0.99;
sigma = 2;
alpha = 0.3;
psi   = 0.5;
phipi = 1.5;
phiy  = 0.25;
rhog  = 0.8;
rhou  = 0.6;

model;
y  = y(+1) - 1/sigma*(r - pi(+1)) + g;
pi = beta*pi(+1) + alpha*psi*y + u;
r  = phipi*pi + phiy*y + er;
g  = rhog*g(-1) + eg;
u  = rhou*u(-1) + eu;
rr = exp(r)/beta;
end;

initval;
rr = 1/0.99;
end;

steady;

shocks;
var eg; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
end;

varobs y pi rr;

estimated_params;
stderr eg, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma_pdf, 0.3, 2;
stderr er, inv_gamma_pdf, 0.2, 2;
beta,  beta_pdf, 0.99, 0.002;
sigma, gamma_pdf, 2, 0.5;
alpha, beta_pdf, 0.3, 0.1;
psi,   beta_pdf, 0.5, 0.1;
phipi, gamma_pdf, 1.5, 0.25;
phiy,  gamma_pdf, 0.25, 0.1;
rhog,  beta_pdf, 0.8, 0.1;
rhou,  beta_pdf, 0.6, 0.1;
end;

identification(checks_via_subsets=1, max_dim_subsets_groups=3, no_identification_strength, nograph);
