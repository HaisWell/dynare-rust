// Small textbook model for the stoch_simul / theoretical-moment checks of dynare-rust
// (reference/analysis/dump_stoch_simul.m). A 3-equation New Keynesian model with an AR(1)
// cost-push shock and correlated demand / policy shocks, plus
//   * a price level p (unit root driven by inflation) and a random walk q (its own shock):
//     the nonstationary variables Dynare drops from the theoretical moments,
//   * dp4 (lags 2 and 3 -> auxiliary lag variables) and ey2 = y(+2) (auxiliary lead variable).
// Run with -DMATLAB_DUMP to save every command's oo_ (verbatim blocks).

var y pi i a p q dp4 ey2;
varexo e_a e_d e_i e_q;
parameters beta sigma kappa phi_pi phi_y rho_a rho_i;

beta = 0.99; sigma = 1.5; kappa = 0.12; phi_pi = 1.6; phi_y = 0.4; rho_a = 0.8; rho_i = 0.6;

model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + e_d;
pi = beta*pi(+1) + kappa*y + a;
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + e_i;
a = rho_a*a(-1) + e_a;
p = p(-1) + pi;
q = q(-1) + 0.5*e_q;
dp4 = pi + pi(-1) + pi(-2) + pi(-3);
ey2 = y(+2);
end;

shocks;
var e_a; stderr 0.5;
var e_d; stderr 0.8;
var e_i; stderr 0.25;
var e_q; stderr 1;
corr e_d, e_i = 0.3;
end;

steady;
check;

stoch_simul(order=1, irf=12, ar=3, conditional_variance_decomposition=[1 4 10], nograph);
@#ifdef MATLAB_DUMP
verbatim;
save('nk_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif

stoch_simul(order=1, irf=8, hp_filter=1600, ar=2, nograph, irf_shocks=(e_a, e_i), relative_irf) y pi i p dp4;
@#ifdef MATLAB_DUMP
verbatim;
save('nk_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif

stoch_simul(order=1, periods=2000, drop=200, ar=3, irf=0, nograph) y pi i p dp4;
@#ifdef MATLAB_DUMP
verbatim;
save('nk_cmd3.mat', 'oo_', 'M_', 'options_');
end;
@#endif

stoch_simul(order=1, periods=1500, hp_filter=1600, irf=0, nograph, nofunctions, nocorr) y pi i;
@#ifdef MATLAB_DUMP
verbatim;
save('nk_cmd4.mat', 'oo_', 'M_', 'options_');
end;
@#endif

stoch_simul(order=1, irf=5, nograph, noprint) y pi;
