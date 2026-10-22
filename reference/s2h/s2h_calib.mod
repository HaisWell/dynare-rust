// smoother2histval with the default invars after calib_smoother (oo_.SmoothedVariables of
// the calibration) on the s2h.mod model, then a forecast from the histval (reference/s2h/dump_s2h.m).
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
calib_smoother(datafile='s2h_data.csv');
smoother2histval;
forecast(periods=8) y z;
