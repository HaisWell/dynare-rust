// MS-SBVAR: sbvar with linear restrictions (`restriction equation`) mixed with exclusions.
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
exclusion lag 0;
equation 1, y3;
restriction equation 1, coeff(y1,0) = 2*coeff(y2,0);
restriction equation 2, coeff(y1,1) + coeff(y2,1) = 0;
exclusion lag 2;
equation 2, y3;
restriction equation 3, 0.5*coeff(y3,2) = coeff(y3,1);
restriction equation 3, - a*coeff(y2,0) = coeff(y1,0) - coeff(y3,0);
end;

sbvar(datafile=msbvar_data, freq=4, initial_year=1960, final_year=1988, nlags=2);
