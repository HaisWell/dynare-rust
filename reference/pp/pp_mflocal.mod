// mod-file local variables take symbol ids (AUX_ENDO_LAG_<symb_id>_<k>) (stream PP)
tmp = 0.5;
var x y;
varexo e;
parameters a;
a = tmp;
model;
x = a*x(-3) + e;
y = x(-2) + 0.1*y(-1);
end;
shocks; var e; stderr 0.01; end;
stoch_simul(order=1, irf=0, nograph);
