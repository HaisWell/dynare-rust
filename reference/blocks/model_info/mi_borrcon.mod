// Consumption-saving model with a borrowing constraint b <= M*y that binds in the steady
// state (BETA*R < 1). The baseline regime is the binding one; the OccBin "bind" regime is
// the one where the constraint is slack (multiplier lb = 0). Positive income shocks make the
// household save and leave the constraint.

var b c lb y;
varexo u;
parameters R BETA M SIGMA RHO;

R     = 1.05;
BETA  = 0.945;
M     = 1;
SIGMA = 2;
RHO   = 0.9;

model;
[name='budget']
c + R*b(-1) = y + b;
[name='borr', relax='borrcon']
b = M*y;
[name='borr', bind='borrcon']
lb = 0;
[name='euler']
lb = c^(-SIGMA) - BETA*R*c(+1)^(-SIGMA);
[name='income']
log(y) = RHO*log(y(-1)) + u;
end;

occbin_constraints;
name 'borrcon'; bind lb < 0; relax b > M*y; error_relax abs(b - M*y);
end;

initval;
y = 1; b = M; c = 1 + M*(1-R); lb = (1+M*(1-R))^(-SIGMA)*(1-BETA*R);
end;

shocks;
var u; stderr 0.05;
end;

shocks(surprise);
var u;
periods 1, 10, 11, 25, 40:41;
values 0.1, -0.05, 0.12, 0.03, 0.08;
end;

occbin_setup(simul_periods=70, simul_check_ahead_periods=60);
occbin_solver;
occbin_write_regimes;
occbin_graph c b lb y u;


model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
