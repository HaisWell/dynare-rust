// perfect-foresight solver variants on the RBC model of reference/deterministic.
// -DCASE selects the perfect_foresight_solver options (see dump_q1pf.m).
@#ifndef CASE
@#define CASE = 1
@#endif
var c k y a;
varexo e;
parameters alpha beta delta rho sigma;
alpha = 0.33;
beta  = 0.99;
delta = 0.025;
rho   = 0.9;
sigma = 2;
model;
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
k = exp(a)*k(-1)^alpha + (1-delta)*k(-1) - c;
y = exp(a)*k(-1)^alpha;
a = rho*a(-1) + e;
end;
initval;
a = 0;
e = 0;
k = 30;
c = 2;
y = 3;
end;
steady;
shocks;
var e; periods 1 4:5; values 0.02 (-rho*0.01);
end;
perfect_foresight_setup(periods=100);
@#if CASE == 1
perfect_foresight_solver(stack_solve_algo=1);
@#elseif CASE == 2
perfect_foresight_solver(stack_solve_algo=2);
@#elseif CASE == 3
perfect_foresight_solver(stack_solve_algo=3, preconditioner=incomplete_lu);
@#elseif CASE == 4
perfect_foresight_solver(stack_solve_algo=2, preconditioner=block_diagonal_lu, block_diagonal_lu_nperiods=10, iter_tol=1e-9);
@#elseif CASE == 5
perfect_foresight_solver(stack_solve_algo=6);
@#elseif CASE == 6
perfect_foresight_solver(stack_solve_algo=7, solve_algo=9);
@#elseif CASE == 7
perfect_foresight_solver(robust_lin_solve);
@#elseif CASE == 8
perfect_foresight_solver(stack_solve_algo=3);
@#elseif CASE == 9
perfect_foresight_solver(stack_solve_algo=7);
@#elseif CASE == 10
perfect_foresight_solver(homotopy_max_completion_share=0.8, homotopy_marginal_linearization_fallback=0.1);
@#endif
