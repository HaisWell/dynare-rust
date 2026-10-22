// VAR(2) with a constant + three var_expectation_models (finite, infinite sum with a parameter in the expression, shifted window)
var z w q1 q2 q3;
varexo e_z e_w e_1 e_2 e_3;
parameters a1 a2 a3 b1 b2 cc beta_ d1;
a1=0.5; a2=0.1; a3=0.15; b1=0.3; b2=0.4; cc=0.05; beta_=0.9; d1=0.2;

var_model(model_name=vv, eqtags=['v_a','v_b']);
var_expectation_model(model_name=ve1, expression=(z+0.5*w), auxiliary_model_name=vv, horizon=3, discount=beta_);
var_expectation_model(model_name=ve2, expression=2*d1*z-w, auxiliary_model_name=vv, horizon=0:Inf, discount=0.8);
var_expectation_model(model_name=ve3, expression=w, auxiliary_model_name=vv, horizon=2:5, discount=beta_, time_shift=-2);

model;
[name='v_a']
z = a1*z(-1) + a2*w(-1) + a3*z(-2) + cc + e_z;
[name='v_b']
w = b1*z(-1) + b2*w(-1) - 0.5*b1*w(-2) + e_w;
q1 = var_expectation(ve1) + e_1;
q2 = 0.5*q2(-1) + var_expectation(ve2) + e_2;
q3 = var_expectation(ve3) + e_3;
end;

shocks;
var e_z; stderr 1;
var e_w; stderr 0.5;
var e_1; stderr 0.2;
var e_2; stderr 0.2;
var e_3; stderr 0.2;
end;

var_expectation.initialize('ve1');
var_expectation.initialize('ve2');
var_expectation.initialize('ve3');
var_expectation.update('ve1');
var_expectation.update('ve2');
var_expectation.update('ve3');
steady;
stoch_simul(order=1, irf=6, nograph, nomoments);
