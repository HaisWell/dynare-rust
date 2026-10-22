// BVAR reference model: three complete columns of the SW07 dataset (reference/sw07/sw07_obs.csv).
var dy pinfobs robs;
varobs dy pinfobs robs;

bvar_density(datafile = sw07_obs, first_obs = 20, bvar_prior_train = 10, bvar_prior_tau = 2.5) 4;
bvar_forecast(forecast = 8, bvar_replic = 200, nobs = 60) 2;
bvar_irf(2, 'SquareRoot');
