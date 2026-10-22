// varexo_det reference (dump_exo_det.m): error: purely backward model
var c a;
varexo e;
varexo_det g;
parameters rho;
rho = 0.9;
model;
c = rho*c(-1) + g + e;
a = 0.5*a(-1) + c;
end;
initval; g = 0.1; end;
steady;
shocks; var e; stderr 0.01; var g; periods 1; values 0.2; end;
stoch_simul(order=1, irf=0);
