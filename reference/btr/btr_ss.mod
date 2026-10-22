// block_trust_region reference: nonlinear steady state from a poor initial guess.
// solve_algo=13 (Dulmage-Mendelsohn blocks); dump.m / the test also run it with 14.
var c k y a z;
varexo e;
parameters alpha beta delta rho sigma;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9; sigma = 2;
model;
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
k = exp(a)*k(-1)^alpha + (1-delta)*k(-1) - c;
y = exp(a)*k(-1)^alpha;
a = rho*a(-1) + e;
z = y + 0.1*c^2 - 1;
end;
initval;
a = 0.3; e = 0.1; k = 80; c = 5; y = 7; z = 0;
end;
steady(solve_algo=13);
