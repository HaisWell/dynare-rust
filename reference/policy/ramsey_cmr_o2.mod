// Christiano, Motto and Rostagno (2007), section 2: `ramsey_policy` at order 2 with a
// planner discount factor (0.98) different from the households' (beta = 0.99), the Ramsey
// steady state solved numerically from initval (no steady-state file), second-order welfare.
var C pi h Z R log_C log_h pi_ann R_ann r_real y_nat y_gap;
varexo epsilon;
parameters beta theta tau chi phi rho;

beta = 0.99;
theta = 5;
phi = 100;
rho = 0.9;
tau = 0;
chi = 1;

model;
[name='Euler equation']
1/(1+R) = beta*C/(C(+1)*pi(+1));
[name='Firm FOC']
(tau-1/(theta-1))*(1-theta)+theta*(chi*h*C/(exp(Z))-1) = phi*(pi-1)*pi-beta*phi*(pi(+1)-1)*pi(+1);
[name='Resource constraint']
C*(1+phi/2*(pi-1)^2) = exp(Z)*h;
[name='TFP process']
Z = rho*Z(-1)+epsilon;
log_C = log(C);
log_h = log(h);
pi_ann = 4*log(pi);
R_ann = 4*R;
r_real = 4*log((1+R)/pi(+1));
y_nat = exp(Z)*sqrt((theta-1)/theta*(1+tau)/chi);
y_gap = log_C-log(y_nat);
end;

initval;
C = 0.9;
pi = 1.001;
h = 0.9;
Z = 0;
R = 0.011;
log_C = log(0.9);
log_h = log(0.9);
pi_ann = 0;
R_ann = 0.04;
r_real = 0.04;
y_nat = 0.9;
y_gap = 0;
end;

shocks;
var epsilon = 0.01^2;
end;

planner_objective log(C)-chi/2*h^2;
ramsey_policy(order=2, instruments=(R), planner_discount=0.98, irf=0) pi_ann log_h R_ann log_C Z r_real;
