function [fval, info, exit_flag, dlik, hess] = rosen5(x, a)
% Rosenbrock as a sum of contributions (for newrat's outer product gradient)
n = length(x);
r = zeros(2*(n-1),1);
for i=1:n-1
    r(2*i-1) = a*(x(i+1)-x(i)^2);
    r(2*i) = 1-x(i);
end
dlik = r.^2;
fval = sum(dlik);
info = zeros(4,1);
exit_flag = 1;
hess = [];
