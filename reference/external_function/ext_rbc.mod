// RBC model whose production function is an external MATLAB function (ces_out.m).
// -DDERIVS=0: numerical derivatives (jacob_element / hess_element)
// -DDERIVS=1: gradient and Hessian returned by ces_out itself (nargout = 2, 3)
// -DDERIVS=2: gradient and Hessian from the separate functions ces_grad.m / ces_hess.m
@#ifndef DERIVS
@#define DERIVS = 1
@#endif
var c k y a;
varexo e;
parameters beta delta alpha rho gam;
beta = 0.99; delta = 0.025; alpha = 0.33; rho = 0.95; gam = 2;

@#if DERIVS == 0
external_function(name = ces_out, nargs = 2);
@#elseif DERIVS == 1
external_function(name = ces_out, nargs = 2, first_deriv_provided, second_deriv_provided);
@#else
external_function(name = ces_out, nargs = 2, first_deriv_provided = ces_grad, second_deriv_provided = ces_hess);
@#endif

model;
y = exp(a)*ces_out(k(-1), alpha);
c^(-gam) = beta*c(+1)^(-gam)*(alpha*exp(a(+1))^(-0.25)*(y(+1)/k)^1.25 + 1 - delta);
k = y - c + (1 - delta)*k(-1);
a = rho*a(-1) + e;
end;

initval;
k = 11.5;
y = 1.9;
c = 1.6;
a = 0;
end;

shocks;
var e; stderr 0.01;
end;

steady;
check;
stoch_simul(order = 2, irf = 0, noprint);
