% dump_postest.m -- Dynare 7.2 post-estimation objects under loglinear and prefilter
% (ref_postest.json), for tests/postest_levels.rs:
%   model_rbc.mod -DSTAGE=4 (loglinear): oo_.forecast of forecast=H (forecasts.run 'smoother'),
%     oo_.shock_decomposition, oo_.forecast_shock_decomposition.unconditional,
%     conditional_forecast at the posterior mode;
%   model_rbc.mod -DSTAGE=5 (loglinear): the forecast command (forecasts.run 'simul'),
%     conditional_forecast(parameter_set=calibration);
%   ../ml/model_ml.mod -DPOSTEST=1 (prefilter=1): forecast=H, shock decompositions,
%     realtime_shock_decomposition, conditional_forecast(parameter_set=calibration);
%   ../ml/model_ml.mod -DPOSTEST=2/3/4: the errors Dynare raises under prefilter (forecast
%     command, conditional_forecast at an estimated parameter set, forecast=H whose var_list
%     misses an observable).
% STAGE 4/5 and POSTEST read the mode files rbc_mode.mat / ../ml/ml_mode.mat written first.
% The conditional forecasts are also recomputed with randn shadowed by zeros (paths without
% free shocks, deterministic).
% Run from a PRIVATE COPY of reference/ (needs ../ml):
%   cd opts; /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_postest.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
global M_ options_ oo_ estim_params_ bayestopt_
zr = fullfile(tempdir, sprintf('zero_randn_%d', feature('getpid')));
mkdir(zr);
fid = fopen(fullfile(zr, 'randn.m'), 'w');
fprintf(fid, 'function x = randn(varargin)\nx = zeros(varargin{:});\n');
fclose(fid);
zflat = @(z) reshape(z, [], 1);
R = struct();

% ---- mode files: the RBC mode (STAGE 6) and the ML mode (no define) -----------------------
cd(here);
dynare model_rbc -DSTAGE=6 nointeractive
copyfile(fullfile(here, 'model_rbc', 'Output', 'model_rbc_mode.mat'), fullfile(here, 'rbc_mode.mat'));
cd(fullfile(here, '..', 'ml'));
dynare model_ml nointeractive
copyfile(fullfile(here, '..', 'ml', 'model_ml', 'Output', 'model_ml_mode.mat'), fullfile(here, '..', 'ml', 'ml_mode.mat'));

% ---- RBC, loglinear, STAGE 4 ------------------------------------------------------------
cd(here);
dynare model_rbc -DSTAGE=4 nointeractive
S = struct();
S.endo_names = M_.endo_names; S.exo_names = M_.exo_names;
S.ys = oo_.dr.ys; S.steady_state = oo_.steady_state;
S.SmoothedVariables = oo_.SmoothedVariables;
S.forecast = oo_.forecast;
S.sd = zflat(oo_.shock_decomposition); S.sd_size = size(oo_.shock_decomposition);
S.sd_fc = zflat(oo_.forecast_shock_decomposition.unconditional);
S.sd_fc_size = size(oo_.forecast_shock_decomposition.unconditional);
S.cf = oo_.conditional_forecast;
addpath(zr, '-begin');
o2 = options_cond_fcst_; o2.replic = 2;
S.cf0 = conditional_forecasts.run(M_, options_, oo_, bayestopt_, estim_params_, constrained_paths_, constrained_vars_, o2);
rmpath(zr);
R.rbc4 = S;

% ---- RBC, loglinear, STAGE 5 ------------------------------------------------------------
dynare model_rbc -DSTAGE=5 nointeractive
S = struct();
S.ys = oo_.dr.ys;
S.forecast = oo_.forecast;
addpath(zr, '-begin');
o2 = options_cond_fcst_; o2.replic = 2;
S.cf0 = conditional_forecasts.run(M_, options_, oo_, bayestopt_, estim_params_, constrained_paths_, constrained_vars_, o2);
rmpath(zr);
R.rbc5 = S;

% ---- ML model, prefilter, POSTEST = 1 ---------------------------------------------------
cd(fullfile(here, '..', 'ml'));
dynare model_ml -DPOSTEST=1 nointeractive
S = struct();
S.endo_names = M_.endo_names; S.exo_names = M_.exo_names;
S.ys = oo_.dr.ys;
[~, dataset_info] = makedataset(options_);
S.mean = dataset_info.descriptive.mean;
S.SmoothedVariables = oo_.SmoothedVariables;
S.forecast = oo_.forecast;
S.sd = zflat(oo_.shock_decomposition); S.sd_size = size(oo_.shock_decomposition);
S.sd_fc = zflat(oo_.forecast_shock_decomposition.unconditional);
S.sd_fc_size = size(oo_.forecast_shock_decomposition.unconditional);
S.rt_pool = zflat(oo_.realtime_shock_decomposition.pool);
S.rt_pool_size = size(oo_.realtime_shock_decomposition.pool);
for t = [150 200]
    f = sprintf('time_%d', t);
    S.(['rt_' f]) = zflat(oo_.realtime_shock_decomposition.(f));
    S.(['rt_' f '_size']) = size(oo_.realtime_shock_decomposition.(f));
    S.(['rtf_' f]) = zflat(oo_.realtime_forecast_shock_decomposition.(f));
    S.(['rtf_' f '_size']) = size(oo_.realtime_forecast_shock_decomposition.(f));
end
addpath(zr, '-begin');
o2 = options_cond_fcst_; o2.replic = 2;
S.cf0 = conditional_forecasts.run(M_, options_, oo_, bayestopt_, estim_params_, constrained_paths_, constrained_vars_, o2);
rmpath(zr);
R.ml1 = S;

% ---- ML model, prefilter: what Dynare rejects --------------------------------------------
for k = 2:4
    msg = '';
    try
        eval(sprintf('dynare model_ml -DPOSTEST=%d nointeractive', k));
    catch err
        msg = err.message;
    end
    R.(sprintf('ml%d_error', k)) = msg;
end
rmdir(zr, 's');

cd(here);
fid = fopen(fullfile(here, 'ref_postest.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dump written');
