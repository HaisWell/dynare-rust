// Christiano, Motto and Rostagno (2007), section 2, as Dynare's
// examples/optimal_policy/nk_ramsey_steady_file.mod but with a `steady_state_model` block
// computing the steady state conditional on the instrument R (dyn_ramsey_static.m with
// steadystate_flag: csolve on the instrument). Second-order Ramsey decision rules and
// second-order (nonlinear) welfare. The labor subsidy tau = 0.1 leaves a distortion.
var C pi h Z R log_C log_h pi_ann R_ann r_real y_nat y_gap;
varexo epsilon;
parameters beta theta tau chi phi rho;

beta = 0.99;
theta = 5;
phi = 100;
rho = 0.9;
tau = 0.1;
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
R = 0.02;
end;

steady_state_model;
Z = 0;
pi = (R+1)*beta;
C = sqrt((1+1/theta*((1-beta)*phi*(pi-1)*pi-(tau-1/(theta-1))*(1-theta)))/(chi*(1+phi/2*(pi-1)^2)));
h = C*(1+phi/2*(pi-1)^2);
log_C = log(C);
log_h = log(h);
pi_ann = 4*log(pi);
R_ann = 4*R;
r_real = 4*log((1+R)/pi);
y_nat = sqrt((theta-1)/theta*(1+tau)/chi);
y_gap = log_C-log(y_nat);
end;

shocks;
var epsilon = 0.01^2;
end;

planner_objective log(C)-chi/2*h^2;
ramsey_model(instruments=(R), planner_discount=beta);
stoch_simul(order=2, irf=0) pi_ann log_h R_ann log_C Z r_real;
evaluate_planner_objective;
