// Stream X test model: resid, model_info, save_params_and_steady_state, parameter changes
// between commands (set_param_value, plain assignment), options_ statements, prior table,
// estimated_params_bounds.
var y c k a ma;
varexo e;
parameters alpha beta delta rho sigma;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.95; sigma = 1;

model;
[name='euler'] 1/c = beta/c(+1)*(alpha*a(+1)*k^(alpha-1) + 1 - delta);
[name='budget'] c + k = a*k(-1)^alpha + (1-delta)*k(-1);
y = a*k(-1)^alpha;
[name='tfp'] log(a) = rho*log(a(-1)) + sigma*e;
ma = (a(-1) + a(-2))/2;
end;

initval;
k = 9; c = 0.7; y = 1; a = 1; ma = 1;
end;

resid;
model_info;
steady;
resid(non_zero);
save_params_and_steady_state('x_misc_ss.txt');

shocks;
var e; stderr 0.01;
end;

options_.nograph = 1;
stoch_simul(order=1, irf=0, noprint);
rho = 0.5;
set_param_value('alpha', 0.3);
stoch_simul(order=1, irf=0, noprint);

estimated_params;
alpha, beta_pdf, 0.35, 0.02;
rho, beta_pdf, 0.9, 0.05;
stderr e, inv_gamma_pdf, 0.01, inf;
delta, gamma_pdf, 0.025, 0.005;
sigma, normal_pdf, 1, 0.1;
beta, uniform_pdf, , , 0.95, 0.995;
end;

estimated_params_bounds;
sigma, 0.5, 1.6;
end;

varobs y;
prior table;
