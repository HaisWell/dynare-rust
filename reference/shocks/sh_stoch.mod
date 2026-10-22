// Stochastic shocks blocks (reference for dynare-rust): variances as expressions, stderr,
// covariances, correlations evaluated at the end of their block, measurement errors,
// shocks(overwrite) (resets Sigma_e and H, not Skew_e), skew / co-skewness entries.
@#ifndef CASE
@#define CASE = 1
@#endif
var c k y;
varexo x z;
parameters alph bet sig;
alph = 0.36; bet = 0.95; sig = 0.01;

model;
  c + k = k(-1)^alph*exp(x) + z;
  1/c = bet/c(+1)*alph*k^(alph-1)*exp(x(+1));
  y = k(-1)^alph*exp(x);
end;

steady_state_model;
  k = (alph*bet)^(1/(1-alph));
  c = k^alph - k;
  y = k^alph;
end;

varobs y c;

shocks;
  var x; stderr 2*sig;
  var z = sig^2;
  corr x, z = 0.5;
  var y; stderr 0.1;
  var c = 0.01;
  corr y, c = 0.2;
  skew z = 0.3;
end;

// the correlation above was evaluated with these variances: changing them keeps the covariance
shocks;
  var x; stderr sig;
  var y = 0.04;
  skew x, x, z = 0.05;
end;

@#if CASE == 2
shocks(overwrite);
  var x; stderr 0.02;
  var z; stderr 0.01;
  var x, z = 0.00005;
  var c; stderr 0.05;
  skew x = -0.5;
end;
@#endif

stoch_simul(order = 1, irf = 0, nomoments, nofunctions);
