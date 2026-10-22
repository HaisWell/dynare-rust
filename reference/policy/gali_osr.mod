// Gali (2015, ch. 5): log-linear New Keynesian model with a cost-push shock and a
// Taylor rule whose coefficients are chosen optimally (OSR) for a quadratic loss.
var pi y_gap i r_nat a u;
varexo eps_a eps_u;
parameters beta sigma varphi alpha epsilon theta rho_a rho_u kappa psi lambda phi_pi phi_y;

beta = 0.99;
sigma = 1;
varphi = 5;
alpha = 0.25;
epsilon = 9;
theta = 0.75;
rho_a = 0.9;
rho_u = 0.5;
kappa = (1-theta)*(1-beta*theta)/theta*(1-alpha)/(1-alpha+alpha*epsilon)*(sigma+(varphi+alpha)/(1-alpha));
psi = (1+varphi)/(sigma*(1-alpha)+varphi+alpha);
lambda = kappa/epsilon;
phi_pi = 1.5;
phi_y = 0.125;

model(linear);
[name='NKPC']
pi = beta*pi(+1) + kappa*y_gap + u;
[name='DIS']
y_gap = y_gap(+1) - 1/sigma*(i - pi(+1) - r_nat);
[name='natural rate']
r_nat = -sigma*psi*(1-rho_a)*a;
[name='Taylor rule']
i = phi_pi*pi + phi_y*y_gap;
a = rho_a*a(-1) + eps_a;
u = rho_u*u(-1) + eps_u;
end;

shocks;
var eps_a; stderr 0.01;
var eps_u; stderr 0.005;
end;

optim_weights;
pi 1;
y_gap lambda;
i 0.05;
pi, y_gap 0.01;
end;

@#ifdef ONLY_PHI_Y
osr_params phi_y;
@#else
osr_params phi_pi phi_y;
@#endif
@#ifndef OPT_ALGO
@#define OPT_ALGO = 4
@#endif
osr(order=1, irf=0, opt_algo=@{OPT_ALGO}) pi y_gap i;
