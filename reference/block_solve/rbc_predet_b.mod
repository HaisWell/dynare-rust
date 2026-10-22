// RBC with Dynare's predetermined_variables timing (k(+1) is the capital chosen in t),
// temporary shock, for the perfect-foresight reference (stream D).
var c k y a;
varexo e;
predetermined_variables k;
parameters alpha beta delta rho sigma;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9; sigma = 2;
model(block);
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*exp(a(+1))*k(+1)^(alpha-1) + 1 - delta);
k(+1) = exp(a)*k^alpha + (1-delta)*k - c;
y = exp(a)*k^alpha;
a = rho*a(-1) + e;
end;
initval;
k = 30; c = 2; y = 3;
end;
steady;
shocks;
var e; periods 1:3; values 0.01;
end;
perfect_foresight_setup(periods=100);
perfect_foresight_solver;
