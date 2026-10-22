% dump.m -- Dynare 7.2: posterior forecasts after an OccBin MCMC (prior_posterior_statistics_core.m,
% OccBin branch: occbin.forecast from the smoothed last state). All kept draws are used
% (sub_draws = number of kept draws), so MeanForecast is deterministic given the draws; the
% draws themselves are dumped so dynare-rust can be fed the same ones (tests/occbin_posterior_forecast.rs).
% Run from a PRIVATE COPY of reference/occbin_fcst:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log.txt 2>&1
% DYNARE_VERSION (environment) selects another Dynare install (default 7.2).
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = pwd;
dynare occbin_fcst -DMCMC nolog nointeractive
x = GetAllPosteriorDraws(options_, M_.dname, M_.fname, 'all');
names = M_.endo_names(1:M_.orig_endo_nbr);
R = struct('draws', x, 'names', {names}, 'mean_forecast_mean', [], 'mean_forecast_median', [], ...
           'mean_forecast_hpdinf', [], 'mean_forecast_hpdsup', [], 'mean_forecast_var', [], 'point_forecast_mean', []);
for i = 1:numel(names)
    n = names{i};
    R.mean_forecast_mean.(n) = oo_.MeanForecast.Mean.(n);
    R.mean_forecast_median.(n) = oo_.MeanForecast.Median.(n);
    R.mean_forecast_hpdinf.(n) = oo_.MeanForecast.HPDinf.(n);
    R.mean_forecast_hpdsup.(n) = oo_.MeanForecast.HPDsup.(n);
    R.mean_forecast_var.(n) = oo_.MeanForecast.Var.(n);
    R.point_forecast_mean.(n) = oo_.PointForecast.Mean.(n);
end
fid = fopen(fullfile(here, 'ref_occbin_fcst.json'), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dump written');
