// MS-SBVAR part 1: constant-parameter SBVAR (sbvar) with exclusion restrictions.
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
equation 3, y1;
exclusion lag 2;
equation 2, y3;
end;

markov_switching(chain=1, number_of_regimes=2, duration=2.5);
markov_switching(chain=2, number_of_regimes=3, duration=2.5, restrictions=[[1,3,0],[3,1,0]]);
svar(variances, chain=1);
svar(coefficients, equations=[1 2], chain=2);

sbvar(datafile=msbvar_data, freq=4, initial_year=1960, final_year=1988, nlags=2, vlistlog=(y1));
