function h = ces_hess(k, alpha)
% Hessian of ces_out with respect to (k, alpha)
% (external_function(..., second_deriv_provided = ces_hess)).
r = -0.25;
kr = k^r;
b = alpha*kr + 1 - alpha;
e1 = b^(1/r - 1);
e2 = b^(1/r - 2);
hkk = alpha*(r - 1)*k^(r - 2)*e1 + alpha^2*(1 - r)*k^(2*r - 2)*e2;
hka = k^(r - 1)*e1 + alpha*k^(r - 1)*(1/r - 1)*e2*(kr - 1);
haa = (kr - 1)^2*(1/r - 1)*e2/r;
h = [hkk, hka; hka, haa];
end
