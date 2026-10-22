// MS-SBVAR: sbvar with linear restrictions only (no exclusion lag), constants excluded.
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
exclusion constants;
restriction equation 2, coeff(y1,0) + coeff(y2,0) + coeff(y3,0) = 0;
restriction equation 1, coeff(y2,1) = 0.3*coeff(y2,2);
restriction equation 3, coeff(y1,0) = 0;
restriction equation 3, 2*coeff(y1,1) - coeff(y3,1) = 0;
end;

sbvar(datafile=msbvar_data, freq=4, initial_year=1960, final_year=1988, nlags=2, coefficients_prior_hyperparameters=[0.8 1.2 0.1 1.2 1.0 1.0]);
