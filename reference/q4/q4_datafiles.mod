// initval_file / histval_file with DATE options (reference/q4/dump_q4.m), on the
// two-lag model of reference/ep/datafiles.mod. CASE 1: initval_file(first_obs=DATE,
// last_obs=DATE); CASE 2: initval_file(first_simulation_period=DATE,
// last_simulation_period=DATE); CASE 3: histval_file(first_obs=DATE).
@#ifndef CASE
@#define CASE = 1
@#endif

var c k;
varexo x;
parameters alph gam delt bet aa;
alph = 0.5; gam = 0.5; delt = 0.02; bet = 0.05; aa = 0.5;

model;
  c + k - aa*x*k(-1)^alph - (1-delt)*k(-1) + 0.1*(k(-1) - k(-2));
  c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;

@#if CASE == 3
initval;
  x = 1;
  k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
  c = aa*k^alph-delt*k;
end;
steady;
@#endif

@#if CASE == 1
initval_file(datafile='ivfd.csv', first_obs=2000Q3, last_obs=2006Q1);
perfect_foresight_setup;
perfect_foresight_solver;
@#elseif CASE == 2
initval_file(datafile='ivfd.csv', first_simulation_period=2001Q1, last_simulation_period=2005Q4);
perfect_foresight_setup;
perfect_foresight_solver;
@#else
histval_file(datafile='ivfd.csv', first_obs=2001Q1, last_obs=2001Q2);
perfect_foresight_setup(periods=30);
perfect_foresight_solver;
@#endif
