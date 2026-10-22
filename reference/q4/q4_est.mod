// reference (reference/q4/dump_q4.m): data command with dates, set_time, ISO
// (daily) dates, epilogue + shock_decomposition(with_epilogue), plot_shock_decomposition
// options. CASE 1: data(first_obs=DATE, last_obs=DATE), epilogue on y only; CASE 2: ISO csv,
// estimation(first_obs=5); CASE 3: epilogue using two endogenous variables (y, c);
// CASE 4: data(file, first_obs=1991Q2, nobs=20) + init2shocks.
@#ifndef CASE
@#define CASE = 1
@#endif
var y c k;
varexo e;
parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;
model;
c + k = exp(y)*k(-1)^alpha + (1-delta)*k(-1);
y = rho*y(-1) + e;
1/c = beta/c(+1)*(alpha*exp(y(+1))*k^(alpha-1)+1-delta);
end;
epilogue;
ya = exp(y)+exp(y(-1))+exp(y(-2))+exp(y(-3));
gya = ya/ya(-4)-1;
@#if CASE == 3
cy = c/exp(y)*alpha;
// Dynare 7.2: begins at max(first(k)+2, first(ya)+0, first(y)+1) (7.1: first(all)+2), and
// y is a prefix of ya (dseries' from ... do matched both before 7.2)
ky = k(-2)/ya + y(-1);
@#endif
end;
steady_state_model;
y = 0; k = ((1/beta-1+delta)/alpha)^(1/(alpha-1)); c = k^alpha - delta*k;
end;
shocks; var e; stderr 0.01; end;
varobs y;
estimated_params;
rho, beta_pdf, 0.9, 0.05;
end;
init2shocks;
y e;
end;
@#if CASE == 1 || CASE == 3
data(file='q4_data.csv', first_obs=1991Q2, last_obs=2000Q4);
estimation(mode_compute=0, smoother, nograph, plot_priors=0);
@#elseif CASE == 2
estimation(datafile='q4_iso.csv', first_obs=5, mode_compute=0, smoother, nograph, plot_priors=0);
@#else
data(file='q4_data.csv', first_obs=1991Q2, nobs=20);
estimation(mode_compute=0, smoother, nograph, plot_priors=0);
@#endif
shock_decomposition(parameter_set=calibration, nograph, with_epilogue) y;
@#if CASE == 4
plot_shock_decomposition(nograph, init2shocks, write_xls, steadystate, fig_name='i2s') y;
@#elseif CASE == 2
plot_shock_decomposition(nograph, write_xls, fig_name='iso') y ya gya;
@#else
plot_shock_decomposition(nograph, write_xls, steadystate, fig_name='w x') y ya gya;
@#endif
