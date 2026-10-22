// MS-SBVAR part 1: sbvar with upper Cholesky identification, non-default hyperparameters.
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
upper_cholesky;
end;

sbvar(datafile=msbvar_data, freq=4, initial_year=1960, final_year=1988, nlags=3, coefficients_prior_hyperparameters=[0.6 0.15 0.5 1.5 1.5 0.8], forecast=2);
