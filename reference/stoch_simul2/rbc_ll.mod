// stoch_simul completion references (reference/stoch_simul2/dump_ss2.m): the RBC
// model (reference/q2/rbc_q2.mod) for `loglinear` at order > 1 (refused by Dynare 7.1) and
// `aim_solver` on a nonlinear model.
// Run with -DCASE=<k>; MATLAB_DUMP saves oo_ after every command.

var y c k i n a g;
varexo ea eg;
parameters alpha beta delta psi rho_a rho_g gbar;

alpha = 0.33; beta = 0.99; delta = 0.025; psi = 1.8; rho_a = 0.95; rho_g = 0.8; gbar = 0.2;

model;
1/c = beta*(1/c(+1))*(alpha*a(+1)*(k/n(+1))^(alpha-1) + 1 - delta);
psi*c/(1-n) = (1-alpha)*a*(k(-1)/n)^alpha;
y = a*k(-1)^alpha*n^(1-alpha);
k = i + (1-delta)*k(-1);
y = c + i + g;
a = 1 - rho_a + rho_a*a(-1) + ea;
log(g) = (1-rho_g)*log(gbar) + rho_g*log(g(-1)) + eg;
end;

steady_state_model;
a = 1;
g = gbar;
n = ((1-alpha)*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha + psi*gbar)/((1-alpha)*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha + psi*((((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha - delta*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))));
k = (((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))*n;
y = k^alpha*n^(1-alpha);
i = delta*k;
c = y - i - g;
end;

shocks;
var ea; stderr 0.01;
var eg; stderr 0.02;
@#if CASE == 9
var c; stderr 0.005;
@#endif
end;
@#if CASE == 9
varobs y c;
@#endif

steady;

@#if CASE == 1
stoch_simul(order=2, loglinear, irf=0, nograph);
@#endif

@#if CASE == 2
stoch_simul(order=2, k_order_solver, loglinear, irf=0, nograph);
@#endif

@#if CASE == 3
stoch_simul(order=3, loglinear, irf=0, nograph);
@#endif

@#if CASE == 4
stoch_simul(order=1, aim_solver, irf=4, ar=1, nograph) y c k;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c4_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 5
stoch_simul(order=2, aim_solver, irf=0, nograph);
@#endif

@#if CASE == 6
// order 2 without k_order_solver: dr=cycle_reduction gives the first-order part
stoch_simul(order=2, dr=cycle_reduction, irf=0, nomoments, nograph) y c k;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c6_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 7
stoch_simul(order=2, k_order_solver, dr=cycle_reduction, irf=0, nograph);
@#endif

@#if CASE == 8
// `order` is reset to 2 by every stoch_simul without the option; print/graph re-enable
stoch_simul(order=1, irf=0, nomoments, noprint, nograph) y;
stoch_simul(print, irf=0, nomoments) y c;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c8_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 9
stoch_simul(order=2, irf=0, ar=1, conditional_variance_decomposition=2, nograph) c n y;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c9_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif
