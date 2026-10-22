// Christiano, Motto and Rostagno (2007), section 2, with a Taylor rule whose coefficients are
// chosen to maximize the second-order unconditional welfare (osr with planner_objective at
// order 2) within osr_params_bounds (opt_algo=9, CMA-ES, in Dynare; the upper bound on phi_pi
// binds).
var C pi h Z R log_C log_h pi_ann R_ann y_gap;
varexo epsilon;
parameters beta theta tau chi phi rho phi_pi phi_y;

beta = 0.99;
theta = 5;
phi = 100;
rho = 0.9;
tau = 0;
chi = 1;
phi_pi = 1.5;
phi_y = 0.1;

model;
1/(1+R) = beta*C/(C(+1)*pi(+1));
(tau-1/(theta-1))*(1-theta)+theta*(chi*h*C/(exp(Z))-1) = phi*(pi-1)*pi-beta*phi*(pi(+1)-1)*pi(+1);
C*(1+phi/2*(pi-1)^2) = exp(Z)*h;
Z = rho*Z(-1)+epsilon;
R = 1/beta-1 + phi_pi*(pi-1) + phi_y*y_gap;
log_C = log(C);
log_h = log(h);
pi_ann = 4*log(pi);
R_ann = 4*R;
y_gap = log_C-log(exp(Z)*sqrt((theta-1)/theta*(1+tau)/chi));
end;

steady_state_model;
Z = 0;
pi = 1;
R = 1/beta-1;
C = sqrt((1-(tau-1/(theta-1))*(1-theta)/theta)/chi);
h = C;
log_C = log(C);
log_h = log(h);
pi_ann = 0;
R_ann = 4*R;
y_gap = log_C-log(sqrt((theta-1)/theta*(1+tau)/chi));
end;

shocks;
var epsilon = 0.01^2;
end;

planner_objective log(C)-chi/2*h^2;
osr_params phi_pi phi_y;
osr_params_bounds;
phi_pi, 1.01, 3;
phi_y, 0, 0.5;
end;
osr(order=2, opt_algo=9, irf=0, planner_discount=0.99) pi_ann R_ann log_C;
