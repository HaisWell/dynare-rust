// shocks / mshocks with dates, vector values and learnt_in dates (reference for dynare-rust).
// CASE 1: perfect_foresight_setup(first_simulation_period = DATE, periods = INTEGER);
// CASE 2: last_simulation_period + periods; CASE 3: first_ and last_simulation_period only;
// CASE 4: perfect foresight with expectation errors, learnt_in given as dates and integers
//         (shocks, mshocks, endval, perfect_foresight_controlled_paths), date periods.
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

xx = [1.02; 1.03; 1.01];
zz = linspace(0.01, 0.03, 3);

shocks;
  var x; periods 2023Q1; values 1.2;
end;

@#if CASE != 4
// overwrite: the first block is dropped
shocks(overwrite);
  var x;
  periods 2023Q1 2023Q2:2023Q4, 5 6:8;
  values 1.1 (xx) (aa*2.1) 1.05;
  var z;
  periods 2:4, 2025Q1;
  values (zz) -0.01;
end;
mshocks;
  var x; periods 2024Q1:2024Q2; values 1.04;
end;
@#else
shocks(overwrite);
  var x; periods 2023Q1:2023Q2; values 1.1;
  var z; periods 2:4; values (zz);
end;
shocks(learnt_in = 2023Q2);
  var x; periods 2023Q3:2023Q4; add 0.05;
  var z; periods 2023Q3 6; values 0.02 0.01;
end;
shocks(learnt_in = 3);
  var x; periods 5; multiply 1.02;
end;
mshocks(learnt_in = 2024Q1);
  var x; periods 2024Q2:2024Q3; values 1.03;
end;
endval(learnt_in = 2023Q3);
  x = 1.1;
  z += 0.01;
end;
endval(learnt_in = 6);
  x *= 0.98;
end;
perfect_foresight_controlled_paths(learnt_in = 2023Q4);
  exogenize c;
  periods 2024Q3:2024Q4;
  values 1.56;
  endogenize z;
end;
@#endif

@#if CASE == 1
perfect_foresight_setup(first_simulation_period = 2023Q1, periods = 40);
perfect_foresight_solver;
@#elseif CASE == 2
perfect_foresight_setup(last_simulation_period = 2032Q4, periods = 40);
perfect_foresight_solver;
@#elseif CASE == 3
perfect_foresight_setup(first_simulation_period = 2023Q1, last_simulation_period = 2032Q4);
perfect_foresight_solver;
@#else
perfect_foresight_with_expectation_errors_setup(first_simulation_period = 2023Q1, periods = 40);
perfect_foresight_with_expectation_errors_solver;
@#endif
