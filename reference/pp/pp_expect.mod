// EXPECTATION and diff() operators (stream PP)
var x y z g;
varexo e;
parameters a b;
a = 0.5; b = 0.8;
model;
x = a*EXPECTATION(-1)(x(+1)) + b*x(-1) + e;
y = EXPECTATION(-1)(y(+1) + x) * 0.5 + 0.2*y(-1) + diff(x);
z = 0.5*diff(z(-1)) + diff(log(1+x)) + 0.3*z(-1) + diff(y(+1));
g = diff(x(-2)) + 0.1*g(-1) + expectation(1)(z(-1));
end;
shocks; var e; stderr 0.01; end;
stoch_simul(order=1, irf=0, nograph);
