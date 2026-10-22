// Dynare 7.2 test fs2000_calib_smoother_cf: conditional_forecast after
// calib_smoother, rebuilt
// as the same scenario on fs2000 with simulated data (fs2000_cf_data.csv, see dump.m).
@#include "fs2000_model.inc"
varobs gp_obs gy_obs;
calib_smoother(datafile='fs2000_cf_data.csv') m P c y;

conditional_forecast_paths;
var gp_obs;
periods 1:2, 3:5;
values 1.012, 1.008;
end;

conditional_forecast(parameter_set=calibration, controlled_varexo=(e_m), periods=10, replic=50);
