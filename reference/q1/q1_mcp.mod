// mixed complementarity problems (lmmcp) — an RBC model with an investment
// irreversibility constraint (i >= 0, complementary to the Euler equation's multiplier
// written as a slack) and a zero lower bound on the policy rate.
// -DCASE=1 perfect_foresight_solver(lmmcp); 2 steady(solve_algo=10) at a binding steady state;
// 3 perfect_foresight_setup + solver(stack_solve_algo=7, solve_algo=10).
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
// Euler equation with the multiplier of the irreversibility constraint
1/c - mu = beta/c(+1)*(alpha*exp(e)*k^(alpha-1) + 1 - delta) - beta*(1-delta)*mu(+1);
c + i = exp(e)*k(-1)^alpha;
k = (1-delta)*k(-1) + i;
mu = 0 ⟂ i > 0;
r = rho*r(-1) + (1-rho)*(1/beta - 1) + 0.5*e _|_ r > 0;
end;
initval;
k = 30; c = 2; i = 0.75; r = 0.0101; mu = 0;
end;
@#if CASE == 2
steady(solve_algo=10);
@#elseif CASE == 4 || CASE == 5
steady;
shocks;
var e; stderr 0.25;
end;
@#if CASE == 4
extended_path(periods=20, lmmcp);
@#else
extended_path(periods=20, lmmcp, order=1);
@#endif
@#else
steady;
shocks;
var e; periods 1:3; values -0.4;
end;
perfect_foresight_setup(periods=60);
@#if CASE == 1
perfect_foresight_solver(lmmcp);
@#else
perfect_foresight_solver(stack_solve_algo=7, solve_algo=10);
@#endif
@#endif
