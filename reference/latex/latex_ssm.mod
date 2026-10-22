// write_latex_steady_state_model reference (Dynare 7.2 preprocessor; 7.1 gives identical files):
// reference/latex/expected/steady_state*.tex come from running
//   /Applications/Dynare/7.2-arm64/preprocessor/dynare-preprocessor latex_ssm.mod
// in a private copy of this directory.
var c $C$ (long_name='consumption') k $K$ y $Y$ z $z$ r_k;
varexo e $\varepsilon$;
parameters alpha $\alpha$ beta $\beta$ delta $\delta$ rho $\rho$ sigma_c $\sigma_c$ kss;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.95;
sigma_c = 2;

model;
c^(-sigma_c) = beta*c(+1)^(-sigma_c)*(r_k(+1) + 1 - delta);
r_k = alpha*exp(z)*k(-1)^(alpha-1);
y = exp(z)*k(-1)^alpha;
k = y - c + (1-delta)*k(-1);
z = rho*z(-1) + 0.5*z(-2) - 0.5*z(-2) + e;
end;

steady_state_model;
z = 0;
r_k = 1/beta - 1 + delta;
kss = (r_k/alpha)^(1/(alpha-1));
k = kss;
y = exp(z)*k^alpha;
c = y - delta*k;
end;

write_latex_steady_state_model;
steady;
