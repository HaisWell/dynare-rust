function f = simpsa_rosen(x)
% Rosenbrock for check_sa_simpsa.m (simpsa takes the objective by name)
x = x(:);
f = sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);
