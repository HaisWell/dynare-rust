// reference model (reference/q2/dump_q2.m): a small RBC model with positive steady
// state (loglinear), all variables present at t (cycle / logarithmic reduction), two shocks.
// Run with -DCASE=<k>; MATLAB_DUMP saves oo_ after every command.

var y c k i n a g;
varexo ea eg;
parameters alpha beta delta psi rho_a rho_g gbar;

alpha = 0.33; beta = 0.99; delta = 0.025; psi = 1.8; rho_a = 0.95; rho_g = 0.8; gbar = 0.2;

model;
1/c = beta*(1/c(+1))*(alpha*a(+1)*(k/n(+1))^(alpha-1) + 1 - delta);
psi*c/(1-n) = (1-alpha)*a*(k(-1)/n)^alpha;
y = a*k(-1)^alpha*n^(1-alpha);
k = i + (1-delta)*k(-1);
y = c + i + g;
a = 1 - rho_a + rho_a*a(-1) + ea;
log(g) = (1-rho_g)*log(gbar) + rho_g*log(g(-1)) + eg;
end;

steady_state_model;
// (no temporary variables: dynare-rust's steady_state_model only assigns model symbols)
a = 1;
g = gbar;
n = ((1-alpha)*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha + psi*gbar)/((1-alpha)*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha + psi*((((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))^alpha - delta*(((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))));
k = (((1/beta - 1 + delta)/alpha)^(1/(alpha-1)))*n;
y = k^alpha*n^(1-alpha);
i = delta*k;
c = y - i - g;
end;

shocks;
var ea; stderr 0.01;
var eg; stderr 0.02;
end;

steady;

@#if CASE == 1
stoch_simul(order=1, loglinear, irf=10, ar=3, conditional_variance_decomposition=[1 4], spectral_density, nograph, tex);
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c1_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=1, loglinear, periods=600, drop=100, irf=0, ar=2, nograph) y c n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c1_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 2
stoch_simul(order=1, bandpass_filter=[6 32], ar=2, irf=0, nograph, contemporaneous_correlation, spectral_density);
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c2_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=1, bandpass_filter, periods=400, drop=50, ar=2, irf=0, nograph) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c2_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 3
stoch_simul(order=1, one_sided_hp_filter=1600, periods=400, drop=50, ar=2, irf=0, nograph, contemporaneous_correlation) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c3_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 8
stoch_simul(order=1, hp_filter=1600, ar=2, irf=0, nograph, spectral_density) y c;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c8_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 4
stoch_simul(order=1, dr=cycle_reduction, irf=6, nograph, nomoments);
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c4_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=1, dr=logarithmic_reduction, dr_logarithmic_reduction_tol=1e-13, dr_logarithmic_reduction_maxiter=200, irf=6, nograph, nomoments);
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c4_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=1, dr=cycle_reduction, dr_cycle_reduction_tol=1e-10, dr_cycle_reduction_maxiter=500, qz_zero_threshold=1e-8, irf=6, nograph, nomoments);
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c4_cmd3.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 5
stoch_simul(order=2, periods=500, drop=100, hp_filter=1600, ar=2, irf=0, nograph, contemporaneous_correlation) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c5_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 6
stoch_simul(order=2, ar=2, irf=0, nograph, conditional_variance_decomposition=[1 4]) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c6_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=2, pruning, ar=2, irf=0, nograph, contemporaneous_correlation, conditional_variance_decomposition=[1 4]) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c6_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(order=3, pruning, ar=2, irf=0, nograph) y c i n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c6_cmd3.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 7
stoch_simul(order=2, pruning, relative_irf, irf=8, replic=2000, nograph, nomoments) y c n;
@#ifdef MATLAB_DUMP
verbatim;
save('q2_c7_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE >= 9
// priors shared by the prior / identification cases (reference/q2/dump_prior.m, dump_ident_q2.m)
varobs y c n;
estimated_params;
alpha, beta_pdf, 0.33, 0.05;
psi, gamma_pdf, 1.8, 0.2;
rho_a, beta_pdf, 0.9, 0.05;
rho_g, beta_pdf, 0.8, 0.1;
stderr ea, inv_gamma_pdf, 0.01, inf;
stderr eg, inv_gamma_pdf, 0.02, inf;
end;
@#endif

@#if CASE == 9
stoch_simul(order=1, irf=8, nograph, noprint, irf_shocks=(ea));
prior moments(distribution) irfs(distribution);
prior optimize;
@#endif

@#if CASE == 10
identification(order=2, analytic_derivation_mode=-1, ar=2, no_identification_strength, nograph);
@#endif

@#if CASE == 11
identification(order=3, analytic_derivation_mode=-1, ar=1, no_identification_strength, nograph);
@#endif

@#if CASE == 12
identification(order=2, ar=2, no_identification_strength, nograph);
@#endif

@#if CASE == 13
identification(checks_via_subsets=1, max_dim_subsets_groups=3, no_identification_strength, nograph);
@#endif
