// trend component model with two ECM equations (lag 2) and two targets, var_expectation_models on it
var y x xs ys q;
varexo e_y e_x e_ys e_xs e_q;
parameters a1 a2 a3 b1 b2 b3 beta_;
a1=-0.3; a2=0.1; a3=0.05; b1=-0.2; b2=0.15; b3=0.02; beta_=0.9;
trend_component_model(model_name=tc, eqtags=['eq_y','eq_x','eq_ys','eq_xs'], targets=['eq_ys','eq_xs']);
var_expectation_model(model_name=ve, expression=(y+0.5*x), auxiliary_model_name=tc, horizon=1:Inf, discount=beta_, time_shift=-1);
var_expectation_model(model_name=ve2, expression=x-xs, auxiliary_model_name=tc, horizon=3, discount=beta_);
model;
[name='eq_y']
diff(y) = a1*(y(-1)-ys(-1)) + a2*diff(y(-1)) + a3*diff(y(-2)) + b3*diff(x(-1)) + e_y;
[name='eq_x']
diff(x) = b1*(x(-1)-xs(-1)) + b2*diff(x(-1)) + a3*diff(y(-1)) + e_x;
[name='eq_ys']
ys = ys(-1) + e_ys;
[name='eq_xs']
xs = xs(-1) + e_xs;
q = var_expectation(ve) + e_q;
end;
var_expectation.initialize('ve');
var_expectation.initialize('ve2');
var_expectation.update('ve');
var_expectation.update('ve2');
