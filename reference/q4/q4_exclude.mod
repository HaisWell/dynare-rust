var y c k z;
varexo e u;
parameters alpha beta delta rho;
alpha=0.36; beta=0.99; delta=0.025; rho=0.95;
model;
[name='res']
c + k = exp(y)*k(-1)^alpha + (1-delta)*k(-1);
[name='tech']
y = rho*y(-1) + e + 0*z;
[name='euler']
1/c = beta/c(+1)*(alpha*exp(y(+1))*k^(alpha-1)+1-delta);
[name='zeq', endogenous='z']
z = 0.5*z(-1) + u;
end;
shocks; var e; stderr 0.01; end;
stoch_simul(order=1, irf=0);
