// purely backward (CASE=1), purely forward (CASE=2) and static (CASE=3) models
// (sim1_purely_backward/forward/static.m).
@#ifndef CASE
@#define CASE = 1
@#endif
var x z w;
varexo e;
parameters rho a;
rho = 0.8;
a = 0.3;
model;
@#if CASE == 1
x = rho*x(-1) + e + a*z(-1)^2;
z = 0.5*z(-1) + 0.2*x;
w = exp(x) + z;
@#elseif CASE == 2
x = rho*x(+1) + e + a*z(+1)^2;
z = 0.5*z(+1) + 0.2*x;
w = exp(x) + z;
@#else
x = e + a*z^2;
z = 0.2*x + 0.1*exp(z);
w = exp(x) + z;
@#endif
end;
initval;
x = 0; z = 0; w = 1; e = 0;
end;
steady;
shocks;
var e; periods 1:2 5; values 0.1 -0.2;
end;
perfect_foresight_setup(periods=20);
perfect_foresight_solver;
