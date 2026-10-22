// RBC with irreversible investment (i >= imin, multiplier mu, complementarity written with
// min()) and government spending: reference model for dynare-rust's extended path.
// CASE selects the extended_path options (see dump_ep.m).
@#ifndef CASE
@#define CASE = 1
@#endif

var c k i a g mu y;
varexo ea eg;
parameters beta alpha delta sigma rho rhog gbar imin;
beta = 0.99; alpha = 0.33; delta = 0.025; sigma = 2; rho = 0.9; rhog = 0.8; gbar = 0.6;
imin = 0.97*delta*((1/beta-1+delta)/alpha)^(1/(alpha-1));

model;
  c^(-sigma) - mu = beta*( c(+1)^(-sigma)*alpha*exp(a(+1))*k^(alpha-1) + (1-delta)*(c(+1)^(-sigma) - mu(+1)) );
  y = exp(a)*k(-1)^alpha;
  k = (1-delta)*k(-1) + i;
  y = c + i + g;
  a = rho*a(-1) + ea;
  log(g/gbar) = rhog*log(g(-1)/gbar) + eg;
  min(mu, i - imin) = 0;
end;

initval;
  k = 28.35; y = 3.015; i = 0.709; c = 1.706; g = 0.6; a = 0; mu = 0;
end;

shocks;
  var ea; stderr 0.02;
  var eg; stderr 0.03;
@#if CASE == 6
  corr ea, eg = 0.3;
@#endif
end;

steady;

@#if CASE == 1
extended_path(periods=60);
@#elseif CASE == 2
extended_path(periods=12, solver_periods=40, order=1, number_of_quadrature_nodes=3);
@#elseif CASE == 3
extended_path(periods=6, solver_periods=25, order=2, number_of_quadrature_nodes=3);
@#elseif CASE == 4
extended_path(periods=12, solver_periods=40, order=2, tree=sparse, number_of_quadrature_nodes=3);
@#elseif CASE == 5
extended_path(periods=12, solver_periods=40, order=1, integration=stroud);
@#elseif CASE == 6
extended_path(periods=12, solver_periods=40, order=2, integration=unscented, tree=sparse);
@#elseif CASE == 7
extended_path(periods=12, solver_periods=40, order=1, number_of_quadrature_nodes=3, hybrid);
@#elseif CASE == 8
extended_path(periods=20, solver_periods=60, use_first_order_solution);
@#endif
