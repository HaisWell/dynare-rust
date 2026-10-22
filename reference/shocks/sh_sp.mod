// shock_paths with dates, databases and expectation errors (reference for dynare-rust).
// CASE 1: dates in periods, dseries / multi-row table / single-row table databases, a
//         controlled stanza with dates, a permanent shock (perfect_foresight_setup);
// CASE 2: integer periods with a table database (no dates);
// CASE 3: expectation errors: shock_paths(learnt_in = INTEGER | DATE), prev. and
//         learnt_in(...) scopes, overwrite (space-separated options), controlled stanzas.
@#ifndef CASE
@#define CASE = 1
@#endif

var c k;
varexo x z w;
parameters alph gam delt bet aa;
alph = 0.5; gam = 0.5; delt = 0.02; bet = 0.05; aa = 0.5;

model;
  c + k - aa*x*k(-1)^alph - (1-delt)*k(-1) + z + w;
  c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;

initval;
  x = 1; z = 0; w = 0;
  k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
  c = aa*k^alph-delt*k;
end;
steady;

db = dseries(transpose(linspace(0, 0.05, 101)), '1999Q3', {'foo'});
tb = table(transpose(linspace(0, 0.1, 101)), 'VariableNames', {'bar'});
flat = array2table([0.004 0.002], 'VariableNames', {'u', 'v'});
database db, tb, flat;

@#if CASE == 1
shock_paths;
  var x;
  periods 2000Q1, 2000Q2:2001Q1, 2001Q2:end;
  values initval.x*1.05, self.x(-1)*1.01 + 0.001, self.x(-1);
  var z;
  periods 2000Q1:2000Q3, 4, 2001Q2;
  values db.foo, tb.bar(1)*0.1, db.foo(-2) + flat.u;
  var w;
  periods 1:3, 9;
  values self.z + flat.v, -self.z(-2);
  exogenize c;
  periods 2002Q2, 2002Q4:2003Q1;
  values initval.c*1.01, 1.55;
  endogenize w;
end;
perfect_foresight_setup(first_simulation_period = 2000Q1, periods = 40);
perfect_foresight_solver;
@#elseif CASE == 2
shock_paths;
  var x;
  periods 1:4, 5:end;
  values 1 + tb.bar, self.x(-1);
  var z;
  periods 2, 3;
  values flat.u, tb.bar(+2) - self.x(-1) + 1;
end;
perfect_foresight_setup(periods = 40);
perfect_foresight_solver;
@#else
shock_paths;
  var x;
  periods 1, 2:5, 6:end;
  values initval.x*1.05, self.x(-1)*1.01, self.x(-1);
  var z;
  periods 2000Q2:2000Q3;
  values db.foo;
  exogenize c;
  periods 8;
  values 1.56;
  endogenize w;
end;
shock_paths(learnt_in = 3);
  // the permanent shock turns out 2% higher than anticipated
  var x;
  periods 6:end;
  values prev.x * 1.02;
  var z;
  periods 3, 4;
  values learnt_in(1).z(-1) + 0.001, prev.x(1) - 1;
end;
shock_paths(learnt_in = 2001Q1);
  var w;
  periods 2001Q1:2001Q2, end;
  values learnt_in(2000Q1).x(1) - 1 + db.foo(+1), 0.002;
  exogenize c;
  periods 2002Q1;
  values 1.555;
  endogenize w;
end;
shock_paths(learnt_in = 7);
  var z; periods 7; values 0.5;
end;
// replaces the previous learnt_in = 7 block (options separated by a space, as the 7.1 grammar)
shock_paths(overwrite learnt_in = 7);
  var z; periods 7, 8; values self.z(-1) + 0.003, prev.z(-1);
end;
perfect_foresight_with_expectation_errors_setup(first_simulation_period = 2000Q1, periods = 40);
perfect_foresight_with_expectation_errors_solver;
@#endif
