% dump_occbin_irf.m -- Dynare 7.2 reference values for occbin.irf (GIRFs from the steady
% state, positive and negative shocks) and occbin.forecast (deterministic, from a given
% initial state with known future shocks) on nk_est.mod, written to ref_occbin_irf.json.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_irf.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_est -DGEN noclearall nograph nolog
global M_ options_ oo_
out = struct();
% GIRFs: a large demand shock (ZLB binds after the negative one), and a supply shock with the
% default size (its standard deviation)
options_.irf = 20;
options_.occbin.irf.exo_names = {'eps_d'; 'eps_s'};
options_.occbin.irf.shocksize = [0.03; 0.002];
irfs = occbin.irf(M_, oo_, options_);
out.irf_piecewise = irfs.piecewise;
out.irf_linear = irfs.linear;
options_.occbin.irf.exo_names = {'eps_m'};
options_.occbin.irf.shocksize = [];
irfs = occbin.irf(M_, oo_, options_);
out.irf2_piecewise = irfs.piecewise;
out.irf2_linear = irfs.linear;
% forecast: initial state below the steady state, known demand shocks in the first periods
M_.endo_histval = oo_.steady_state;
M_.endo_histval(strcmp(M_.endo_names, 'r_nat')) = -0.012;
M_.endo_histval(strcmp(M_.endo_names, 'i')) = 0.004;
options_.occbin.forecast.SHOCKS0 = {{'eps_d', [-0.01 -0.005 zeros(1, 13)]}};
options_.occbin.forecast.waitbar = false;
[f, error_flag] = occbin.forecast(options_, M_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 15);
out.forecast_error_flag = error_flag;
out.forecast_piecewise = f.piecewise.Mean;
out.forecast_linear = f.linear.Mean;
out.forecast_regime = arrayfun(@(x) double(x.regime), f.regimes, 'UniformOutput', false);
out.forecast_regimestart = arrayfun(@(x) double(x.regimestart), f.regimes, 'UniformOutput', false);
out.endo_histval = M_.endo_histval;
fid = fopen(fullfile(here, 'ref_occbin_irf.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin_irf.json');
