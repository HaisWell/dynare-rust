// smoother2histval with the default invars on a model with an AR(2) lag (AUX_ENDO_LAG) and a
// lagged shock (AUX_EXO_LAG) after an estimation at fixed parameters, then a forecast from the histval (reference/s2h/dump_s2h.m).
var y z;
varexo e u;
parameters rho1 rho2 a rz;
rho1 = 0.6;
rho2 = 0.2;
a = 0.8;
rz = 0.5;

model;
y = rho1*y(-1) + rho2*y(-2) + a*u(-1) + e;
z = rz*z(-1) + u;
end;

shocks;
var e; stderr 0.1;
var u; stderr 0.05;
end;

steady;
varobs y z;
estimated_params;
rho1, 0.6;
end;
// mode_compute = 0: no optimization, the smoother at the initial values (deterministic)
estimation(datafile='s2h_data.csv', mode_compute=0, smoother, nograph);
smoother2histval;
forecast(periods=8) y z;
