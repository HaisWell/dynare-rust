// stoch_simul completion references (reference/stoch_simul2/dump_ss2.m): a small New Keynesian
// model with persistent demand / cost-push shocks and interest-rate smoothing, for
// `partial_information` (dr1_PI / PI_gensys / PCL_Part_info_irf / PCL_Part_info_moments) and
// `aim_solver` / `dr=aim` (AIM_first_order_solver / SPAmalg).
// Run with -DCASE=<k>; MATLAB_DUMP saves oo_ after every command.

var y pi r v u;
varexo ev eu er;
parameters beta sigma kappa phi_pi phi_y rho_r rho_v rho_u;

beta = 0.99; sigma = 1; kappa = 0.15; phi_pi = 1.5; phi_y = 0.25; rho_r = 0.6; rho_v = 0.8; rho_u = 0.5;

@#if CASE == 9
model;
@#else
model(linear);
@#endif
y = y(+1) - (1/sigma)*(r - pi(+1)) + v;
pi = beta*pi(+1) + kappa*y + u;
r = rho_r*r(-1) + (1-rho_r)*(phi_pi*pi + phi_y*y) + er;
v = rho_v*v(-1) + ev;
u = rho_u*u(-1) + eu;
end;

shocks;
var ev; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
@#if CASE == 4
corr ev, eu = 0.4;
@#endif
@#if CASE == 10
var pi; stderr 0.25;
var y; stderr 0.4;
@#endif
end;

@#if CASE == 1 || CASE == 3 || CASE == 4 || CASE == 6 || CASE == 7 || CASE == 10
varobs y pi;
@#endif

@#if CASE == 1
stoch_simul(partial_information, irf=8, ar=3, nograph);
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c1_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(irf=4, ar=2, nograph, nofunctions) y pi;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c1_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 2
stoch_simul(partial_information, order=2, irf=6, ar=2, nograph, contemporaneous_correlation) y r;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c2_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 3
stoch_simul(partial_information, nocorr, ar=0, irf=5, nograph, periods=200) pi r u;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c3_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 4
stoch_simul(partial_information, nomoments, relative_irf, irf=6, irf_shocks=(ev, er), nograph);
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c4_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(partial_information, irf=0, ar=1, nograph);
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c4_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 5
stoch_simul(aim_solver, order=1, irf=6, ar=2, nograph);
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c5_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(dr=aim, order=1, irf=4, nomoments, nograph) y pi;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c5_cmd2.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 6
stoch_simul(partial_information, noprint, irf=4, nograph);
@#endif

@#if CASE == 7
stoch_simul(partial_information, nocorr, irf=0, nograph);
@#endif

@#if CASE == 8
stoch_simul(partial_information, spectral_density, irf=0, ar=1, nograph) y;
@#endif

@#if CASE == 9
// declared nonlinear, Hessian equal to zero: order 2 falls back to order 1
stoch_simul(order=2, irf=3, ar=1, nograph) y pi;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c9_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
@#endif

@#if CASE == 10
// measurement errors on observed variables (M_.H): MODEL SUMMARY, unconditional and
// conditional variance decompositions WITH MEASUREMENT ERROR
stoch_simul(irf=0, ar=1, conditional_variance_decomposition=[1 4], nograph) pi r y;
@#ifdef MATLAB_DUMP
verbatim;
save('ss2_c10_cmd1.mat', 'oo_', 'M_', 'options_');
end;
@#endif
stoch_simul(periods=300, drop=50, irf=0, ar=1, nograph) pi y;
@#endif

@#if CASE == 11
// tex with graph_format=eps: the <fname>_IRF.tex eps-loader
stoch_simul(tex, graph_format=eps, irf=4, nomoments, nofunctions, nomodelsummary);
@#endif
