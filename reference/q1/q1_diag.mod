// model_diagnostics.m output.
// CASE=1 no problem; 2 static Jacobian singular (a redundant equation in the static model:
// Walras law with a variable that only enters in first differences); 3 unit root (static
// singularity from a random walk); 4 uncalibrated parameter, the steady state fails.
@#ifndef CASE
@#define CASE = 1
@#endif
var y c k a w;
varexo e;
parameters alpha beta delta rho gam;
alpha = 0.33; beta = 0.99; delta = 0.025;
@#if CASE != 4
rho = 0.9;
@#endif
gam = 0.5;
model;
1/c = beta/c(+1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
y = exp(a)*k(-1)^alpha;
k = y - c + (1-delta)*k(-1);
@#if CASE == 2
c + k = y + (1-delta)*k(-1) + w - w(-1);
@#elseif CASE == 3
w = w(-1) + e;
@#elseif CASE == 4
w = log(gam*a);
@#else
w = (1-alpha)*y;
@#endif
a = rho*a(-1) + e;
end;
@#if CASE == 2 || CASE == 3
steady_state_model;
a = 0;
k = ((1/beta-1+delta)/alpha)^(1/(alpha-1));
y = k^alpha;
c = y - delta*k;
w = 1;
end;
@#else
initval;
k = 30; c = 2; y = 3; a = 0; w = 2;
end;
@#endif
model_diagnostics;
