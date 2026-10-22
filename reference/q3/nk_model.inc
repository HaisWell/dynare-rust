var y $y$ (long_name='output gap') pi $\pi$ (long_name='inflation') i $i$ (long_name='nominal_rate');
varexo e_y $\varepsilon_y$ (long_name='demand shock') e_pi $\varepsilon_\pi$ e_i;
parameters rho_y $\rho_y$ (long_name='output persistence') beta $\beta$ kappa $\kappa$ phi_pi $\phi_\pi$ (long_name='Taylor 100% rule') rho_i;
rho_y = 0.7;
beta = 0.99;
kappa = 0.1;
phi_pi = 1.5;
rho_i = 0.6;

model(linear);
# r = i - pi(+1);
[name='IS curve']
y = rho_y*y(-1) + 0.1*y(-2) - 0.1*r + e_y;
[name='Phillips curve', mcp='pi>0']
pi = 0.5*beta*pi(+1) + 0.4*pi(-1) + kappa*y + e_pi;
[name='Taylor rule']
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + 0.5*y) + e_i;
end;

shocks;
var e_y; stderr 1;
var e_pi; stderr 0.5;
var e_i; stderr 0.25;
end;
