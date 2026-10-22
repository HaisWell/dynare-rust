// constant equations (DynamicModel::simplifyEquations) and predetermined_variables (stream PP)
var k c y z pi;
varexo e;
parameters alpha beta delta;
predetermined_variables k;
alpha = 0.36; beta = 0.99; delta = 0.025;
model;
z = 1;
pi = 0.5;
y = z*k^alpha*exp(e);
k(+1) = y - c + (1-delta)*k;
1/c = beta/c(+1)*(alpha*z(+1)*y(+1)/k(+1) + 1 - delta) + pi(-1) - 0.5;
end;
initval; k = 30; c = 2; y = 3; z = 1; pi = 0.5; end;
shocks; var e; stderr 0.01; end;
steady;
stoch_simul(order=1, irf=0, nograph);
