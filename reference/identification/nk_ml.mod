// nk_ident.mod estimated by maximum likelihood (estimated_params without priors): Dynare
// identifies at the ML starting values ("ML_Starting_value"), whatever parameter_set says,
// with one point only (prior_mc is reset to 1). Stream R.
// Small New Keynesian model for the identification tests of dynare-rust.
// The Phillips-curve slope is the product alpha*psi: the pair (alpha, psi) is
// not identified (only the product enters the model), the rest is.
// The gross rate rr = exp(r)/beta makes the model nonlinear and gives a
// parameter-dependent steady state (tests dYss and the g2 term of dg1).

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
stderr eg, 0.45, 0.01, 5;
stderr eu, 0.35, 0.01, 5;
stderr er, 0.25, 0.01, 5;
beta,  0.985, 0.9, 0.999;
sigma, 1.8, 0.5, 5;
alpha, 0.35, 0.05, 0.95;
psi,   0.45, 0.05, 0.95;
phipi, 1.6, 1.01, 3;
phiy,  0.2, 0.01, 1;
rhog,  0.75, 0.01, 0.99;
rhou,  0.65, 0.01, 0.99;
end;

identification(parameter_set=prior_mean, prior_mc=10, advanced=1);
