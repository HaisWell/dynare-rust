// Multiplicative deterministic shocks (mshocks) on a nonzero exogenous steady state,
// exogenous leads/lags (aux variables in a deterministic model), for stream D.
var y c;
varexo g;
parameters a b;
a = 0.5; b = 0.3;
model;
y = a*y(-1) + b*c(+1) + g + 0.2*g(-1) + 0.1*g(+1);
c = 0.9*c(-1) + 0.1*y;
end;
initval;
g = 1;
y = 3;
c = 3;
end;
steady;
mshocks;
var g; periods 1:2 6; values 1.1 (a+0.5);
end;
perfect_foresight_setup(periods=40);
perfect_foresight_solver;
