@#ifndef ORDER
@#define ORDER = 3
@#endif
// Nonlinear New Keynesian model with Rotemberg pricing, preference, technology and monetary
// shocks and a two-lag interest-rate rule (exercises aux variables at higher order; stream K).
var c pi r y n w a v d;
varexo ea ev ed;
parameters beta sigma phi epsilon kappa pibar rhor phipi phiy rhoa rhov rhod;
beta = 0.995; sigma = 1.5; phi = 1; epsilon = 6; kappa = 50; pibar = 1.005;
rhor = 0.7; phipi = 1.5; phiy = 0.25; rhoa = 0.9; rhov = 0.5; rhod = 0.8;
model;
c^(-sigma) = beta*exp(d(+1)-d)*c(+1)^(-sigma)*r/pi(+1);
w = n^phi*c^sigma;
y = exp(a)*n;
(1-epsilon) + epsilon*w/exp(a) - kappa*(pi/pibar-1)*pi/pibar
  + beta*exp(d(+1)-d)*(c(+1)/c)^(-sigma)*kappa*(pi(+1)/pibar-1)*pi(+1)/pibar*y(+1)/y = 0;
y = c + kappa/2*(pi/pibar-1)^2*y;
log(r) = (1-rhor)*log(pibar/beta) + rhor*(0.6*log(r(-1)) + 0.4*log(r(-2)))
  + (1-rhor)*(phipi*log(pi/pibar) + phiy*log(y/y(-1))) + v;
a = rhoa*a(-1) + ea;
v = rhov*v(-1) + ev;
d = rhod*d(-1) + ed;
end;
initval;
pi = 1.005; r = 1.0100502512562815; w = 0.8333333333333334;
n = 0.9296671847748563; y = 0.9296671847748563; c = 0.9296671847748563;
a = 0; v = 0; d = 0;
end;
shocks;
var ea; stderr 0.01;
var ev; stderr 0.0025;
var ed; stderr 0.01;
end;


model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
stoch_simul(order=1, irf=0, noprint, nomoments, nocorr, nofunctions, nograph);
