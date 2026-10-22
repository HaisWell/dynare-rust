// lead/lag aux variables in a stochastic model, model-local variables with leads (stream PP)
var x y z w;
varexo e u;
parameters a b c;
a = 0.5; b = 0.9; c = 0.2;
model;
# q = x(+1)*y;
# s = q + c*z(-1);
x = a*x(+3) + b*x(-1) + e(+2) + 0.1*u(-2);
y = q(+1) + 0.5*exp(y(+2)) - 0.5 + c*y(-3) + normcdf(x(+1)) - 0.5 + erf(z(+2))/10;
z = b*z(-1) + s(-1)*0.01 + max(w(+2), 0)*0.01 + abs(e(+1))*sign(u) + u;
w = -x(+2)*z(+1) + 0.3*w(-1) + 1/(2+y(+2))*0.1 - 0.05 + e*e(+1);
end;
initval; y = 0; end;
shocks; var e; stderr 0.01; var u; stderr 0.01; end;
stoch_simul(order=1, irf=0, nograph);
