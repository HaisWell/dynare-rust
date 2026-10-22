// VAR auxiliary model (stationary levels), simple target (no composite), kind inferred (ll)
var x y z w;
varexo ex ey ez ew;
parameters a_x_1 a_x_2 b_x_1 a_y_1 b_y_1 beta e_c_m c_z_1 c_z_2;
a_x_1 = .5; a_x_2 = -.1; b_x_1 = .15; a_y_1 = .3; b_y_1 = .2;
beta = .95; e_c_m = .2; c_z_1 = .3; c_z_2 = -.1;
var_model(model_name=toto, eqtags=['eq:x', 'eq:y']);
pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman);
model;
[name='eq:x']
x = a_x_1*x(-1) + a_x_2*x(-2) + b_x_1*y(-1) + ex;
[name='eq:y']
y = a_y_1*y(-1) + b_y_1*x(-1) + ey;
[name='zpac']
diff(z) = e_c_m*(x(-1)-z(-1)) + c_z_1*diff(z(-1)) + c_z_2*diff(z(-2)) + pac_expectation(pacman) + ez;
w = .5*w(-1) + z + ew;
end;
shocks;
var ex = .01; var ey = .02; var ez = .015; var ew = .01;
end;
pac.initialize('pacman');
pac.update.expectation('pacman');
steady;
stoch_simul(order=1, irf=8, nograph, nomoments);
