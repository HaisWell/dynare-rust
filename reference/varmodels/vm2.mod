// structural VAR(2) without constant + var_expectation (horizon 4:Inf, shifted) in the model
var z w q;
varexo e_z e_w e_q;
parameters d1 g1 g2 beta_;
d1=0.2; g1=0.3; g2=0.4; beta_=0.95;
var_model(model_name=vv, eqtags=['v_a','v_b'], structural);
var_expectation_model(model_name=ve, expression=z-2*w, auxiliary_model_name=vv, horizon=4:Inf, discount=beta_, time_shift=-1);
model;
[name='v_a']
z = g1*w + d1*w(-1) + 0.1*z(-2) + 0.3*z(-1) + e_w;
[name='v_b']
w = g2*z + 0.5*d1*z(-1) + 0.5*w(-1) + e_z;
[name='v_q']
q = var_expectation(ve) + e_q;
end;
shocks; var e_z; stderr 1; var e_q; stderr .3; var e_w; stderr .1; end;
var_expectation.initialize('ve');
var_expectation.update('ve');
steady;
stoch_simul(order=1, irf=6, nograph, nomoments);
