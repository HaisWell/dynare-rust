// initval_file / histval_file reference for dynare-rust (two lags: one auxiliary variable).
// CASE 1: initval_file (csv), periods from the file; CASE 2: initval_file (xlsx, first_obs,
// last_obs) + perfect_foresight_setup(periods=20); CASE 3: histval_file + perfect foresight;
// CASE 4: histval_file + extended_path.
// Dynare 7.2 histval_file semantics (exactly M_.orig_maximum_lag observations are read):
// CASE 5: a long file, no option (the first two rows); CASE 6: first_obs only; CASE 7:
// last_obs only. Error cases (no simulation): CASE 8: first_obs/last_obs span differs from
// the maximum lag; CASE 9: too few observations before last_obs; CASE 10: too few after
// first_obs; CASE 11/12: the removed nobs / last_simulation_period options (parse errors);
// CASE 13: initval_file with nobs, which Dynare 7.2 rejects with a failed assertion (its
// preprocessor passes the caller 'INITVALF', the new asserts test 'INITVAL').
@#ifndef CASE
@#define CASE = 1
@#endif

var c k;
varexo x;
parameters alph gam delt bet aa;
alph = 0.5; gam = 0.5; delt = 0.02; bet = 0.05; aa = 0.5;

model;
@#if CASE <= 3 || CASE >= 5
  c + k - aa*x*k(-1)^alph - (1-delt)*k(-1) + 0.1*(k(-1) - k(-2));
  c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
@#else
  // stochastic context: no exogenous lead (Dynare would substitute the whole expectation)
  c + k - aa*exp(x)*k(-1)^alph - (1-delt)*k(-1) + 0.1*(k(-1) - k(-2));
  c^(-gam) - (1+bet)^(-1)*(aa*alph*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
@#endif
end;

@#if CASE >= 3 && CASE <= 7
initval;
@#if CASE == 4
  x = 0;
@#else
  x = 1;
@#endif
  k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
  c = aa*k^alph-delt*k;
end;
steady;
@#endif

@#if CASE == 1
initval_file(datafile='ivf.csv');
perfect_foresight_setup;
perfect_foresight_solver;
@#elseif CASE == 2
initval_file(datafile='ivf.xlsx', first_obs=3, last_obs=26);
perfect_foresight_setup(periods=20);
perfect_foresight_solver;
@#elseif CASE == 3
histval_file(datafile='hvf.csv');
perfect_foresight_setup(periods=40);
perfect_foresight_solver;
@#elseif CASE == 4
shocks; var x; stderr 0.01; end;
histval_file(datafile='hvf.csv');
extended_path(periods=8, solver_periods=40);
@#elseif CASE <= 7
@#if CASE == 5
histval_file(datafile='ivf.csv');
@#elseif CASE == 6
histval_file(datafile='ivf.csv', first_obs=10);
@#else
histval_file(datafile='ivf.csv', last_obs=20);
@#endif
perfect_foresight_setup(periods=40);
perfect_foresight_solver;
@#elseif CASE == 8
histval_file(datafile='ivf.csv', first_obs=3, last_obs=5);
@#elseif CASE == 9
histval_file(datafile='ivf.csv', last_obs=1);
@#elseif CASE == 10
histval_file(datafile='ivf.csv', first_obs=33);
@#elseif CASE == 11
histval_file(datafile='ivf.csv', nobs=2);
@#elseif CASE == 12
histval_file(datafile='ivf.csv', first_obs=1, last_simulation_period=5);
@#else
initval_file(datafile='ivf.csv', first_obs=3, nobs=24);
@#endif
