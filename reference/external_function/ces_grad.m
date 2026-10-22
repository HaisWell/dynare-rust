function g = ces_grad(k, alpha)
% Gradient of ces_out with respect to (k, alpha), as a separate function
% (external_function(..., first_deriv_provided = ces_grad)).
r = -0.25;
kr = k^r;
b = alpha*kr + 1 - alpha;
e1 = b^(1/r - 1);
g = [alpha*k^(r - 1)*e1, (kr - 1)*e1/r];
end
