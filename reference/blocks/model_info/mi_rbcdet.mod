// Ramsey/RBC model for the perfect-foresight reference (dynare-rust stream D).
// -DCASE=1 permanent technology shock (endval + steady)
//        2 temporary shocks (shocks block, two period groups)
//        3 temporary shocks, closed-form steady_state_model
//        4 temporary shocks, linear_approximation
//        5 histval start away from the steady state, endogenous_terminal_period
//        6 permanent shock solved by the homotopy loop (homotopy_initial_step_size=0.5)
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
@#if CASE == 3
steady_state_model;
a = e/(1-rho);
k = ((1/beta-1+delta)/(alpha*exp(a)))^(1/(alpha-1));
y = exp(a)*k^alpha;
c = y - delta*k;
end;
@#else
initval;
a = 0;
e = 0;
k = 30;
c = 2;
y = 3;
end;
@#endif
@#if CASE == 1 || CASE == 6
endval;
e = 0.01;
end;
@#if CASE == 6
@#else
@#endif
@#elseif CASE == 5
histval;
k(0) = 25;
a(0) = 0.05;
end;
@#else
shocks;
var e; periods 1 4:5; values 0.02 (-rho*0.01);
end;
@#if CASE == 4
@#else
@#endif
@#endif


model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
perfect_foresight_setup(periods=5);
