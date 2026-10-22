@#ifndef ORDER
@#define ORDER = 3
@#endif
// RBC with CRRA utility, endogenous labour, TFP and government spending shocks
// (reference model for higher-order perturbation, stream K).
var y c k l i z g;
varexo ez eg;
parameters beta alpha delta sigma psi rho rhog gbar;
beta = 0.99; alpha = 0.33; delta = 0.025; sigma = 2; psi = 2;
rho = 0.95; rhog = 0.9; gbar = 0.2;
model;
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*exp(z(+1))*k^(alpha-1)*l(+1)^(1-alpha) + 1 - delta);
psi*c^sigma/(1-l) = (1-alpha)*exp(z)*k(-1)^alpha*l^(-alpha);
y = exp(z)*k(-1)^alpha*l^(1-alpha);
k = i + (1-delta)*k(-1);
y = c + i + g;
z = rho*z(-1) + ez;
g = (1-rhog)*gbar + rhog*g(-1) + eg;
end;
initval;
l = 0.4188697641102882; k = 11.874295605000956; y = 1.2630296059803614;
i = 0.29685739012502393; c = 0.7661722158553375; z = 0; g = 0.2;
end;
steady;
shocks;
var ez; stderr 0.01;
var eg; stderr 0.005;
var ez, eg = 0.3*0.01*0.005;
end;
stoch_simul(order=@{ORDER}, irf=0, nograph, noprint);
