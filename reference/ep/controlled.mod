// perfect_foresight_controlled_paths reference for dynare-rust (the manual's example).
// CASE 1: Newton (sim1); CASE 2: linear_approximation (sim1_linear); CASE 3: homotopy
// (homotopy_initial_step_size=0.5: the controlled values are scaled by the share);
// CASE 4: with expectation errors (learnt_in).
@#ifndef CASE
@#define CASE = 1
@#endif

var c k;
varexo x z;
parameters alph gam delt bet aa;
alph = 0.5; gam = 0.5; delt = 0.02; bet = 0.05; aa = 0.5;

model;
  c + k - aa*x*k(-1)^alph - (1-delt)*k(-1) + z;
  c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;

initval;
  x = 1; z = 0;
  k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
  c = aa*k^alph-delt*k;
end;
steady;

shocks;
  var x; periods 1; values 1.2;
end;

perfect_foresight_controlled_paths;
  exogenize c;
  periods 2 4:5;
  values 1.6 (1.04*1.6);
  endogenize x;

  exogenize k;
  periods 7:9;
  values 13;
  endogenize z;
end;

@#if CASE == 4
perfect_foresight_controlled_paths(learnt_in=3);
  exogenize c;
  periods 4:5;
  values 1.7;
  endogenize x;
end;
perfect_foresight_with_expectation_errors_setup(periods=60);
perfect_foresight_with_expectation_errors_solver;
@#else
perfect_foresight_setup(periods=60);
@#if CASE == 1
perfect_foresight_solver;
@#elseif CASE == 2
perfect_foresight_solver(linear_approximation);
@#else
perfect_foresight_solver(homotopy_initial_step_size=0.5);
@#endif
@#endif
