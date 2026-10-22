// Estimated stderrs of two shocks while the shocks block calibrates their covariance (STAGE 1)
// or correlation (STAGE 2); STAGE 3: a calibrated covariance survives when another corr is estimated;
// STAGE 4: a calibrated measurement-error covariance survives with only ME stderrs estimated (ncn = 0). See check_for_calibrated_covariances.m / set_all_parameters.m.
@#ifndef STAGE
@#define STAGE = 1
@#endif
var c k a y;
@#if STAGE == 3
varexo e u v;
@#else
varexo e u;
@#endif
parameters alpha beta delta rho;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;
model;
1/c = beta/c(+1)*(alpha*a(+1)*k^(alpha-1) + 1 - delta);
c + k = y + (1-delta)*k(-1);
y = a*k(-1)^alpha;
@#if STAGE == 3
log(a) = rho*log(a(-1)) + e + u + v;
@#else
log(a) = rho*log(a(-1)) + e + u;
@#endif
end;
initval;
a = 1; k = 30; c = 2.3; y = 3;
end;
steady;
shocks;
var e; stderr 0.01;
var u; stderr 0.005;
@#if STAGE == 1 || STAGE == 3
var e, u = 0.00002;
@#else
corr e, u = 0.4;
@#endif
@#if STAGE == 3
var v; stderr 0.004;
corr u, v = 0.2;
@#endif
var y; stderr 0.004;
@#if STAGE == 4
var c; stderr 0.003;
var c, y = 0.000006;
@#endif
end;
varobs c y;
estimated_params;
rho, 0.85, beta_pdf, 0.8, 0.1;
stderr e, 0.012, inv_gamma_pdf, 0.01, inf;
stderr u, 0.006, inv_gamma_pdf, 0.01, inf;
@#if STAGE == 3
corr u, v, 0.3, uniform_pdf, , , -0.9, 0.9;
@#endif
@#if STAGE == 4
stderr y, 0.005, inv_gamma_pdf, 0.01, inf;
@#endif
end;
estimation(datafile=rbc_levels, loglinear, order=1, mode_compute=0, nograph, nodisplay);
