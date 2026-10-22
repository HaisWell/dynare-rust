function [y, dy, d2y] = ces_out(k, alpha)
% CES production with labour normalised to 1 and a fixed elasticity of substitution:
%   y = (alpha*k^r + 1 - alpha)^(1/r),  r = (sigma - 1)/sigma.
% Optional outputs: gradient and Hessian with respect to (k, alpha)
% (external_function(..., first_deriv_provided, second_deriv_provided)).
sigma = 0.8;
r = (sigma - 1)/sigma;
if abs(r) < 1e-12
    % Cobb-Douglas limit (not used with sigma = 0.8)
    y = k^alpha;
else
    b = alpha*k^r + 1 - alpha;
    y = b^(1/r);
end
if nargout > 1
    kr = k^r;
    b = alpha*kr + 1 - alpha;
    e1 = b^(1/r - 1);
    dy = zeros(1, 2);
    dy(1) = alpha*k^(r - 1)*e1;
    dy(2) = (kr - 1)*e1/r;
end
if nargout > 2
    e2 = b^(1/r - 2);
    d2y = zeros(2, 2);
    d2y(1, 1) = alpha*(r - 1)*k^(r - 2)*e1 + alpha^2*(1 - r)*k^(2*r - 2)*e2;
    d2y(1, 2) = k^(r - 1)*e1 + alpha*k^(r - 1)*(1/r - 1)*e2*(kr - 1);
    d2y(2, 1) = d2y(1, 2);
    d2y(2, 2) = (kr - 1)^2*(1/r - 1)*e2/r;
end
end
