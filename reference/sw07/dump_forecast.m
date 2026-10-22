% dump_forecast.m -- reference values for the Rust `forecast` module (ref_forecast.json).
%
% STAGE 9 run (mode file, mh_replic=0, smoother, filtered_vars, forecast=12): Dynare's
% point forecast at the mode from the last smoothed state (forecasts.run(...,'smoother'),
% forecasts.stochastic_classical: mean and conf_sig bands), plus the `forecast` command's
% 'simul' task (start from the steady state, periods=8) at the same decision rule.
%
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_forecast.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

R = struct();
R.var_list = var_list_;
R.conf_sig = options_.forecasts.conf_sig;
R.horizon = options_.forecast;
R.forecast = oo_.forecast;              % Mean / HPDinf / HPDsup per variable (horizon x 1)
% last smoothed values of every endogenous variable (incl. auxiliary ones): forecast start
last = zeros(M_.endo_nbr, 1);
for i = 1:M_.endo_nbr
    v = oo_.SmoothedVariables.(M_.endo_names{i});
    last(i) = v(end);
end
R.last_smoothed = last;
R.endo_names = M_.endo_names;

% `forecast(periods=8)` command at the same parameters (task 'simul': from the steady state)
opts = options_;
opts.periods = 8;
opts.nograph = true;
opts.forecasts.conf_sig = 0.8;
R.simul_conf_sig = 0.8;
R.forecast_simul = forecasts.run(var_list_, M_, opts, oo_, 'simul');

fid = fopen('ref_forecast.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_forecast.json written\n');
