// Stochastic extended path with complementarity conditions solved by lmmcp
// (options_.ep.solve_algo = 10, set in dump.m after the dynare run: Dynare's preprocessor has no
// option for it). Same model as reference/q1/q1_mcp.mod.
// -DCASE=1 order=1 sparse tree; 2 order=2, sparse tree, hybrid=2; 3 order=0 with ep.solve_algo=10.
@#ifndef CASE
@#define CASE = 1
@#endif
var c k i r mu;
varexo e;
parameters alpha beta delta rho rbar;
alpha = 0.36;
beta  = 0.99;
delta = 0.025;
rho   = 0.8;
rbar  = 0.01;
model;
1/c - mu = beta/c(+1)*(alpha*exp(e)*k^(alpha-1) + 1 - delta) - beta*(1-delta)*mu(+1);
c + i = exp(e)*k(-1)^alpha;
k = (1-delta)*k(-1) + i;
mu = 0 ⟂ i > 0;
r = rho*r(-1) + (1-rho)*(1/beta - 1) + 0.5*e _|_ r > 0;
end;
initval;
k = 30; c = 2; i = 0.75; r = 0.0101; mu = 0;
end;
steady;
shocks;
var e; stderr 0.25;
end;
@#if CASE == 1
extended_path(periods=12, solver_periods=40, order=1, tree=sparse, number_of_quadrature_nodes=3);
@#elseif CASE == 2
extended_path(periods=10, solver_periods=30, order=2, tree=sparse, number_of_quadrature_nodes=3, hybrid=2);
@#else
extended_path(periods=20);
@#endif
