// steady printout reference (stream R): disp_steady_state.m pads the names to the longest
// endogenous name, auxiliary variables included (the 16-character AUX_ENDO_ names here). steady_print.log is
// Dynare 7.1's log of `dynare steady_print nointeractive` (run from a private copy).
var c k a y z;
varexo e;
parameters alpha beta delta rho;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;

model;
1/c = beta/c(+1)*(alpha*a(+1)*k^(alpha-1) + 1 - delta);
c + k = y + (1-delta)*k(-1);
y = a*k(-1)^alpha;
log(a) = rho*log(a(-1)) + e;
z = y(-3) + c(+2) - 1e-7*k;
end;

initval;
a = 1; k = 30; c = 2.3; y = 3; z = 0;
end;
steady;
steady;
