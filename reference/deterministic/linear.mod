// Small New Keynesian model(linear) for the perfect-foresight reference (stream D):
// sim1_linear path, a lead of 2 (aux variable), the legacy `simul` command.
var y pi i;
varexo e_i e_y;
parameters beta kappa phi_pi rho_i;
beta = 0.99; kappa = 0.1; phi_pi = 1.5; rho_i = 0.7;
model(linear);
y = y(+1) - (i - pi(+1)) + e_y;
pi = beta*pi(+2)*0.5 + beta*0.5*pi(+1) + kappa*y;
i = rho_i*i(-1) + (1-rho_i)*phi_pi*pi + e_i;
end;
steady;
shocks;
var e_i; periods 1:4; values 0.25;
var e_y; periods 2; values -0.5;
end;
simul(periods=60);
