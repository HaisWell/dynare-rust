// [static]/[dynamic] equations and OccBin bind/relax snippets (stream PP)
var c y r k i;
varexo e;
parameters beta alpha delta rstar phi ilb;
beta = 0.99; alpha = 0.33; delta = 0.025; rstar = 1/0.99; phi = 1.5; ilb = 0.02;
model;
[name='euler']
1/c = beta*r/c(+1);
[dynamic]
y = exp(e)*k(-1)^alpha;
[static]
y = k^alpha;
k = (1-delta)*k(-1) + i;
c + i = y;
[name='mp', relax='zlb']
r = rstar + phi*(y - STEADY_STATE(y));
[name='mp', bind='zlb']
r = 1;
end;
occbin_constraints;
name 'zlb'; bind r <= 1; relax r > 1;
end;
steady_state_model;
r = rstar;
k = ((1/beta-1+delta)/alpha)^(1/(alpha-1));
y = k^alpha;
i = delta*k;
c = y - i;
end;
shocks; var e; stderr 0.01; end;
steady;
stoch_simul(order=1, irf=0, nograph);
