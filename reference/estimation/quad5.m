function [fval, info, exit_flag, dlik, hess] = quad5(x, A, b)
% f = sum over rows k of 0.5*(A(k,:)*x - b(k))^2
r = A*x - b;
dlik = 0.5*r.^2;
fval = sum(dlik);
info = zeros(4,1);
exit_flag = 1;
hess = [];
