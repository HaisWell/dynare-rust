% dump_conditional.m -- reference values for stream C (ref_conditional.json):
% conditional forecasts (conditional_forecasts.run / get_shock_paths), realtime and
% initial-condition shock decompositions, plot_shock_decomposition transformations and
% shock_decomposition's forecast_type.
%
% Runs tests/data/sw07_conditional.mod (copy it next to sw07.mod first): the SW07 STAGE 9
% estimation (mode file) followed by the scenario's commands. Then, at the same mode:
%   - the conditional / unconditional paths without free shocks, through conditional_forecasts.run
%     itself with randn shadowed by zeros (deterministic: Rust must match to rounding);
%   - the same with parameter_set = calibration (start from the steady state);
%   - realtime_shock_decomposition pools and vintages, also with fast_realtime;
%   - initial_condition_decomposition;
%   - plot_shock_decomposition(..., get_decomp_only) for several option sets;
%   - shock_decomposition with forecast_type = conditional / unconditional.
%
% Run from a private copy of reference/sw07 (with sw07_conditional.mod copied in):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_conditional.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
copyfile(fullfile(here, '..', '..', 'tests', 'data', 'sw07_conditional.mod'), here);
dynare sw07_conditional nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

vv = {'pinf4'; 'y'; 'r'};
iv = zeros(numel(vv), 1);
for k = 1:numel(vv)
    iv(k) = find(strcmp(vv{k}, M_.endo_names));
end
flat = @(z) reshape(z(iv, :, :), [], 1);
sz = @(z) size(z(iv, :, :));

R = struct();
R.vars = vv;
R.endo_names = M_.endo_names;
R.exo_names = M_.exo_names;
R.nobs = options_.nobs;
[dataset_, ~] = makedataset(options_);
R.firstdate = char(dataset_.firstdate);
R.constrained_vars = constrained_vars_;
R.constrained_paths = constrained_paths_;
R.cf = oo_.conditional_forecast;            % Monte Carlo (MATLAB's randn), replic = 5000

% ---- deterministic paths: randn -> zeros ---------------------------------------------
zr = fullfile(tempdir, sprintf('zero_randn_%d', feature('getpid')));
mkdir(zr);
fid = fopen(fullfile(zr, 'randn.m'), 'w');
fprintf(fid, 'function x = randn(varargin)\nx = zeros(varargin{:});\n');
fclose(fid);
addpath(zr, '-begin');
o2 = options_cond_fcst_;
o2.replic = 2;
R.cf0 = conditional_forecasts.run(M_, options_, oo_, bayestopt_, estim_params_, constrained_paths_, constrained_vars_, o2);
o3 = o2;
o3.parameter_set = 'calibration';
R.cf0_calibration = conditional_forecasts.run(M_, options_, oo_, bayestopt_, estim_params_, constrained_paths_, constrained_vars_, o3);
rmpath(zr);
rmdir(zr, 's');
R.params_after_estimation = M_.params;

% ---- realtime shock decomposition (from the scenario: forecast=4, save_realtime=[80 86 90])
rt = struct();
rt.pool = flat(oo_.realtime_shock_decomposition.pool);
rt.pool_size = sz(oo_.realtime_shock_decomposition.pool);
for t = [80 86 90]
    f = sprintf('time_%d', t);
    rt.(f) = flat(oo_.realtime_shock_decomposition.(f));
    rt.([f '_size']) = sz(oo_.realtime_shock_decomposition.(f));
    rt.(['forecast_' f]) = flat(oo_.realtime_forecast_shock_decomposition.(f));
    rt.(['forecast_' f '_size']) = sz(oo_.realtime_forecast_shock_decomposition.(f));
end
for t = [80 86]
    f = sprintf('time_%d', t);
    rt.(['conditional_' f]) = flat(oo_.conditional_shock_decomposition.(f));
    rt.(['conditional_' f '_size']) = sz(oo_.conditional_shock_decomposition.(f));
    rt.(['realtime_conditional_' f]) = flat(oo_.realtime_conditional_shock_decomposition.(f));
    rt.(['realtime_conditional_' f '_size']) = sz(oo_.realtime_conditional_shock_decomposition.(f));
end
rt.conditional_pool = flat(oo_.conditional_shock_decomposition.pool);
rt.conditional_pool_size = sz(oo_.conditional_shock_decomposition.pool);
rt.forecast_pool = flat(oo_.realtime_forecast_shock_decomposition.pool);
rt.forecast_pool_size = sz(oo_.realtime_forecast_shock_decomposition.pool);
R.realtime = rt;
R.realtime_parameter_set = oo_.shock_decomposition_info.realtime_shock_decomposition.parameter_set;

% ---- initial condition decomposition (scenario command) -----------------------------
R.initval = flat(oo_.initval_decomposition);
R.initval_size = sz(oo_.initval_decomposition);

% ---- plot_shock_decomposition transformations (get_decomp_only) ---------------------
oo_base = oo_;
opts = options_;
opts.parameter_set = 'posterior_mode';
opts.no_graph.shock_decomposition = true;
oo_sd = shock_decomposition(M_, oo_base, opts, vv, bayestopt_, estim_params_);
cases = { ...
    struct('name', 'rt1_v86_yoy_flip_main', 'realtime', 1, 'vintage', 86, 'type', 'yoy', 'flip', true,  'groups', 'main', 'diff', false), ...
    struct('name', 'rt3_v86',               'realtime', 3, 'vintage', 86, 'type', '',    'flip', false, 'groups', '',     'diff', false), ...
    struct('name', 'rt2_v86',               'realtime', 2, 'vintage', 86, 'type', '',    'flip', false, 'groups', '',     'diff', false), ...
    struct('name', 'rt2_pool',              'realtime', 2, 'vintage', 0,  'type', '',    'flip', false, 'groups', 'main', 'diff', false), ...
    struct('name', 'rt1_pool_diff',         'realtime', 1, 'vintage', 0,  'type', 'qoq', 'flip', false, 'groups', '',     'diff', true), ...
    struct('name', 'rt0_aoa',               'realtime', 0, 'vintage', 0,  'type', 'aoa', 'flip', false, 'groups', '',     'diff', false), ...
    struct('name', 'rt0_aoa_dated',         'realtime', 0, 'vintage', 0,  'type', 'aoa', 'flip', false, 'groups', '',     'diff', false), ...
    struct('name', 'rt0_yoy_main',          'realtime', 0, 'vintage', 0,  'type', 'yoy', 'flip', true,  'groups', 'main', 'diff', false)};
P = struct();
for k = 1:numel(cases)
    c = cases{k};
    o = set_default_plot_shock_decomposition_options(options_);
    o.plot_shock_decomp.realtime = c.realtime;
    o.plot_shock_decomp.vintage = c.vintage;
    o.plot_shock_decomp.type = c.type;
    o.plot_shock_decomp.flip = c.flip;
    o.plot_shock_decomp.diff = c.diff;
    o.plot_shock_decomp.use_shock_groups = c.groups;
    if strcmp(c.name, 'rt0_aoa_dated')
        o.initial_date = dataset_.firstdate;   % as when shock_decomposition itself plots
    end
    try
        zz = plot_shock_decomposition(M_, oo_sd, o, vv, true);
        P.(c.name) = reshape(zz, [], 1);
        P.([c.name '_size']) = size(zz);
    catch e
        P.(c.name) = e.message;
    end
end
R.plot = P;

% ---- fast_realtime ---------------------------------------------------------------------
o = options_;
o.parameter_set = 'posterior_mode';
o.shock_decomp.forecast = 0;
o.shock_decomp.save_realtime = [70 85];
o.shock_decomp.fast_realtime = [60 80];
oo_f = rmfield(oo_base, intersect(fieldnames(oo_base), {'realtime_shock_decomposition', 'realtime_forecast_shock_decomposition', ...
    'realtime_conditional_shock_decomposition', 'conditional_shock_decomposition'}));
oo_f = realtime_shock_decomposition(M_, oo_f, o, vv, bayestopt_, estim_params_);
R.fast = struct('pool', flat(oo_f.realtime_shock_decomposition.pool), 'pool_size', sz(oo_f.realtime_shock_decomposition.pool), ...
    'time_70', flat(oo_f.realtime_shock_decomposition.time_70), 'time_70_size', sz(oo_f.realtime_shock_decomposition.time_70), ...
    'time_85', flat(oo_f.realtime_shock_decomposition.time_85), 'time_85_size', sz(oo_f.realtime_shock_decomposition.time_85), ...
    'conditional_pool', flat(oo_f.conditional_shock_decomposition.pool));

% ---- shock_decomposition with forecast_type ---------------------------------------------
F = struct();
for ft = {'conditional', 'unconditional'}
    for ps = {'posterior_mode', 'calibration'}
        o = options_;
        o.parameter_set = ps{1};
        o.no_graph.shock_decomposition = true;
        o.shock_decomp.forecast_type = ft{1};
        name = [ft{1} '_' ps{1}];
        try
            oo_t = shock_decomposition(M_, oo_base, o, vv, bayestopt_, estim_params_);
            z = oo_t.forecast_shock_decomposition.(ft{1});
            F.(name) = flat(z);
            F.([name '_size']) = sz(z);
        catch e
            F.(name) = e.message;
        end
    end
end
R.forecast_type = F;
R.forecast_mean = oo_.forecast.Mean;   % estimation's forecast at the mode (uf for calibration)

fid = fopen('ref_conditional.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_conditional.json written\n');
