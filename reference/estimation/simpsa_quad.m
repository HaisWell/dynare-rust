function f = simpsa_quad(x)
% Shifted quadratic (minimum 3) for check_sa_simpsa.m
f = 3 + sum([1;10;0.5;4].*(x(:)-[1.5;-2;0.5;3]).^2);
