// Christiano, Motto and Rostagno (2007), section 2 (as in Dynare's
// examples/optimal_policy/nk_ramsey_osr.mod): New Keynesian model with Rotemberg
// price adjustment costs, Ramsey policy with the agents' utility as planner objective.
// Variant with two instruments (the interest rate R and the labor subsidy tau, which the
// planner uses to undo the monopolistic distortion) and leads/lags of two periods, so that
// the multipliers get auxiliary variables. No steady-state file. The instruments are declared
// last: with several instruments, dyn_ramsey_static.m measures the FOC residual by the last
// row of a QR factor, which needs the FOCs of the other variables to be independent (with R
// and tau declared in the middle, Dynare and dynare-rust both fail to find the steady state).
var C pi h Z log_C log_h pi_ann R_ann r_real y_nat y_gap pi_fwd2 dZ2 R tau;
varexo epsilon;
parameters beta theta chi phi rho tau_bar;

beta = 0.99;
theta = 5;
phi = 100;
rho = 0.9;
tau_bar = 0.1;
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
y_nat = exp(Z)*sqrt((theta-1)/theta*(1+tau_bar)/chi);
y_gap = log_C-log(y_nat);
pi_fwd2 = log(pi(+2));
dZ2 = Z - Z(-2);
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
tau = 0.2;
pi_fwd2 = 0;
dZ2 = 0;
end;

shocks;
var epsilon = 0.01^2;
end;

planner_objective log(C)-chi/2*h^2-(tau-tau_bar)^2;
ramsey_model(instruments=(R, tau), planner_discount=beta);
stoch_simul(order=1, irf=0) pi_ann log_h R_ann log_C Z r_real tau dZ2;
evaluate_planner_objective;
