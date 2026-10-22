// nk_ident.mod without an estimated_params block (identification then looks at all stderr
// and model parameters at their current values) and with an extra state z that does not
// reach the observables: the state space is not minimal (Komunjer-Ng goes through minreal),
// rhoz does not move any moment, and alpha/psi are again only identified as a product.
var y pi r g u rr z;
varexo eg eu er;
parameters beta sigma alpha psi phipi phiy rhog rhou rhoz;

beta  = 0.99;
sigma = 2;
alpha = 0.3;
psi   = 0.5;
phipi = 1.5;
phiy  = 0.25;
rhog  = 0.8;
rhou  = 0.6;
rhoz  = 0.5;

model;
y  = y(+1) - 1/sigma*(r - pi(+1)) + g;
pi = beta*pi(+1) + alpha*psi*y + u;
r  = phipi*pi + phiy*y + er;
g  = rhog*g(-1) + eg;
u  = rhou*u(-1) + eu;
rr = exp(r)/beta;
z  = rhoz*z(-1) + eg;
end;

initval;
rr = 1/0.99;
end;

steady;

shocks;
var eg; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
end;

varobs y pi rr;

identification(advanced=1);
