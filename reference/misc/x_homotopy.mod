// Stream X test model: steady(homotopy_mode=MODE) with a homotopy_setup block moving a
// parameter (with an explicit start value) and an exogenous variable (from its initval).
@#ifndef MODE
@#define MODE = 1
@#endif
var y c k;
varexo x;
parameters alpha beta delta;
alpha = 0.36; beta = 0.99; delta = 0.025;
model;
1/c = beta/c(+1)*(alpha*exp(x)*k^(alpha-1) + 1 - delta);
c + k = exp(x)*k(-1)^alpha + (1-delta)*k(-1);
y = exp(x)*k(-1)^alpha;
end;
initval;
k = 10; c = 1; y = 1.5; x = 0.05;
end;
homotopy_setup;
alpha, 0.3, 0.36;
x, 0.1;
end;
steady(homotopy_mode=@{MODE}, homotopy_steps=4);
