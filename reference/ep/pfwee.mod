// Perfect foresight with expectation errors (reference for dynare-rust's ep::pfwee).
// CASE 1: shocks/mshocks/endval with learnt_in (values, add, multiply, +=, *=, overwrite);
// CASE 2: the same with constant_simulation_length; CASE 3: information from a CSV datafile;
// CASE 4: a regular endval block (permanent shock) revised by endval(learnt_in=...).
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

@#if CASE == 4
endval;
  x = 1.05;
end;
steady;
@#endif

@#if CASE != 3
shocks;
  var x; periods 1:2; values 1.1;
end;
shocks(learnt_in=2);
  var x; periods 3:4; add 0.05;
  var z; periods 2 3; values 0.02 0.01;
end;
shocks(learnt_in=3);
  var z; periods 4; values 0.5;
end;
shocks(learnt_in=3, overwrite);
  var x; periods 5; multiply 1.02;
end;
mshocks(learnt_in=4);
  var x; periods 6:7; values 1.03;
end;
endval(learnt_in=3);
  x = 1.1;
  z += 0.01;
end;
endval(learnt_in=6);
  x *= 0.98;
end;
@#endif

@#if CASE == 1 || CASE == 4
perfect_foresight_with_expectation_errors_setup(periods=60);
perfect_foresight_with_expectation_errors_solver;
@#elseif CASE == 2
perfect_foresight_with_expectation_errors_setup(periods=60);
perfect_foresight_with_expectation_errors_solver(constant_simulation_length);
@#else
perfect_foresight_with_expectation_errors_setup(periods=7, datafile='pfwee_data.csv');
perfect_foresight_with_expectation_errors_solver;
@#endif
