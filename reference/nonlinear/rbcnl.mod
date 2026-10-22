@#ifndef ORDER
@#define ORDER = 2
@#endif
// RBC of reference/korder/rbc.mod (uncorrelated shocks) observed with measurement errors on
// y and c: reference model for nonlinear estimation with particle filters (stream PF).
// With -DSIMULATE, only simulates (dump_nonlinear.m writes rbcnl_data.csv from it).
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
varobs y c;
shocks;
var ez; stderr 0.02;
var eg; stderr 0.01;
var y; stderr 0.004;
var c; stderr 0.003;
end;
@#ifdef SIMULATE
stoch_simul(order=2, periods=300, drop=0, irf=0, nograph, noprint, nomoments);
@#else
estimated_params;
rho, 0.95, beta_pdf, 0.9, 0.05;
sigma, 2, gamma_pdf, 2, 0.5;
stderr ez, 0.02, inv_gamma_pdf, 0.02, 2;
stderr y, 0.004, inv_gamma_pdf, 0.005, 2;
stderr c, 0.003, inv_gamma_pdf, 0.005, 2;
end;
@#if defined(ESTGF)
// mode of the (deterministic) Gaussian filter with cubature rules
estimation(datafile='rbcnl_data.csv', order=@{ORDER}, filter_algorithm=gf, proposal_approximation=cubature,
           distribution_approximation=cubature, mode_compute=4, mh_replic=0, nograph, nodisplay);
@#elseif defined(ESTSIS)
// RW Metropolis-Hastings with the bootstrap particle filter, from the gf mode
estimation(datafile='rbcnl_data.csv', order=@{ORDER}, filter_algorithm=sis, number_of_particles=1000,
           mode_compute=0, mode_file='rbcnl/Output/rbcnl_mode.mat', mh_replic=2000, mh_nblocks=2, mh_jscale=0.5,
           nograph, nodisplay, nodiagnostic);
@#else
estimation(datafile='rbcnl_data.csv', order=@{ORDER}, filter_algorithm=sis, number_of_particles=200,
           mode_compute=0, mh_replic=0, nograph, nodisplay);
@#endif
@#endif
