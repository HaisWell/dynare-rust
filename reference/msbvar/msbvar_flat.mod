// MS-SBVAR part 1: sbvar without Bayesian prior, lower Cholesky.
var y1 y2 y3;
varexo e1;
parameters a;
a = 0.5;
model;
y1 = a*y1(-1) + e1;
y2 = y1;
y3 = y2;
end;
varobs y1 y2 y3;

svar_identification;
lower_cholesky;
end;

sbvar(datafile=msbvar_data, freq=4, initial_year=1960, final_year=1988, nlags=2, no_bayesian_prior);
