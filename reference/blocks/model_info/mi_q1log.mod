var(log) y k;
var c z;
varexo e;
parameters alpha beta delta rho;
alpha=0.36; beta=0.99; delta=0.025; rho=0.9;
model;
c + k = y + (1-delta)*k(-1) + 0*z(-2);
y = exp(e)*k(-1)^alpha;
1/c = beta/c(+1)*(alpha*y(+2)/k + 1 - delta);
z = rho*z(-1) + e + diff(log(y)) + y(-3);
end;
initval;
k = 10; y = 2; c = 1;
end;
shocks; var e; stderr 0.01; end;


model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
stoch_simul(order=1, irf=0, noprint, nomoments, nocorr, nofunctions, nograph);
