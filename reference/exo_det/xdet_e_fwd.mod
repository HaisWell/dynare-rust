// varexo_det reference (dump_exo_det.m): error: purely forward model
var c a;
varexo e;
varexo_det g;
parameters beta rho;
beta = 0.99; rho = 0.9;
model;
c = beta*c(+1) + g + e;
a = rho*a(+1) + c;
end;
initval; g = 0.1; end;
steady;
shocks; var e; stderr 0.01; var g; periods 1; values 0.2; end;
stoch_simul(order=1, irf=0);
