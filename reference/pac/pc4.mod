// model consistent expectations (no auxiliary model), stationary target
var x z y;
varexo ex ez ey;
parameters rho beta e_c_m c_z_1 c_z_2 gg;
rho=.7; beta=.9; e_c_m=.1; c_z_1=.7; c_z_2=-.3; gg=0;
pac_model(discount=beta, model_name=pacman, growth=gg);
model;
x = rho*x(-1) + ex;
[name='zpac']
diff(z) = e_c_m*(x(-1)-z(-1)) + c_z_1*diff(z(-1))  + c_z_2*diff(z(-2)) + pac_expectation(pacman) + ez;
y = .5*y(-1) + z + ey;
end;
shocks; var ex=.01; var ez=.02; var ey=.01; end;
pac.initialize('pacman');
pac.mce.parameters('pacman');
steady;
stoch_simul(order=1, irf=8, nograph, nomoments);
