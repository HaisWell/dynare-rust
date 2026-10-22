var x p q;
varexo e;
parameters b k;
b=0.9; k=0.2;
model;
x = x(+1) - 0.5*(q - p(+1));
p = b*p(+1) + k*x + e;
exp(q) = exp(1.5*p);
end;


model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
perfect_foresight_setup(periods=5);
