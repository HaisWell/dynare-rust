// Gali (2015, ch. 5): log-linear New Keynesian model with a cost-push shock,
// optimal discretionary policy for the quadratic loss pi^2 + lambda*y_gap^2.
var pi y_gap i r_nat a u;
varexo eps_a eps_u;
parameters beta sigma varphi alpha epsilon theta rho_a rho_u kappa psi lambda;

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

model(linear);
[name='NKPC']
pi = beta*pi(+1) + kappa*y_gap + u;
[name='DIS']
y_gap = y_gap(+1) - 1/sigma*(i - pi(+1) - r_nat);
[name='natural rate']
r_nat = -sigma*psi*(1-rho_a)*a;
a = rho_a*a(-1) + eps_a;
u = rho_u*u(-1) + eps_u;
end;

shocks;
var eps_a; stderr 0.01;
var eps_u; stderr 0.005;
end;

planner_objective pi^2 + lambda*y_gap^2;
discretionary_policy(instruments=(i), irf=0, planner_discount=beta) pi y_gap i;
