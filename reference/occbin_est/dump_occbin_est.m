% dump_occbin_est.m -- Dynare 7.2 reference values for OccBin estimation (tests/occbin_est.rs):
% the piecewise Kalman filter likelihood (dsge_likelihood with options_.occbin.likelihood),
% with and without filter_use_relaxation, the inversion-filter likelihood (occbin.IVF_posterior),
% at several parameter points, and the OccBin smoother (occbin.DSGE_smoother: smoothed
% variables, shocks and regimes; IVF smoother) at the initial values. Written to
% ref_occbin_est.json.
%
% Run from a PRIVATE COPY of this folder (Dynare writes +nk_est/ and nk_est/ next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_est.m')"
%
% For debugging only, a copy of missing_observations_kalman_filter.m that also stores the
% filter's regimes (global OCCBIN_PKF_DEBUG) is written to this folder; it computes exactly
% what Dynare's does.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
src = fileread(fullfile(droot, 'matlab/kalman/likelihood/missing_observations_kalman_filter.m'));
src = strrep(src, sprintf('%% Divide by two.\n'), sprintf(['global OCCBIN_PKF_DEBUG\nif occbin_.status, OCCBIN_PKF_DEBUG = struct(''regimes'', regimes_, ''a0'', a0, ''a1'', a1, ''lik'', lik, ''t'', t); end\n' ...
    '%% Divide by two.\n']));
fid = fopen('missing_observations_kalman_filter.m', 'w'); fprintf(fid, '%s', src); fclose(fid);
rehash;
global OCCBIN_PKF_DEBUG

variants = { '', 'pkf'; '-DRELAX', 'relax'; '-DIVF', 'ivf' };
out = struct();
for v = 1:size(variants, 1)
    clear functions
    if isempty(variants{v, 1})
        dynare nk_est noclearall nograph nolog
    else
        eval(sprintf('dynare nk_est %s noclearall nograph nolog', variants{v, 1}));
    end
    global M_ options_ oo_ estim_params_ bayestopt_
    R = struct();
    % smoother of the estimation command (mode_compute=0: initial values)
    if isfield(oo_, 'occbin') && isfield(oo_.occbin, 'smoother')
        s = oo_.occbin.smoother;
        sm = struct();
        f = {'alphahat', 'etahat', 'error_flag', 'warning_flag'};
        for k = 1:numel(f)
            if isfield(s, f{k}), sm.(f{k}) = s.(f{k}); end
        end
        if isfield(s, 'regime_history') && ~isempty(s.regime_history)
            sm.regime = arrayfun(@(x) double(x.regime), s.regime_history, 'UniformOutput', false);
            sm.regimestart = arrayfun(@(x) double(x.regimestart), s.regime_history, 'UniformOutput', false);
        end
        if isfield(s, 'realtime_regime_history') && ~isempty(s.realtime_regime_history)
            sm.realtime_regime = arrayfun(@(x) double(x.regime), s.realtime_regime_history, 'UniformOutput', false);
            sm.realtime_regimestart = arrayfun(@(x) double(x.regimestart), s.realtime_regime_history, 'UniformOutput', false);
        end
        if isfield(s, 'SmoothedVariables')
            sm.SmoothedVariables = s.SmoothedVariables;
            sm.SmoothedShocks = s.SmoothedShocks;
        end
        if isfield(oo_.occbin, 'linear_smoother')
            sm.linear_alphahat = oo_.occbin.linear_smoother.alphahat;
            sm.linear_etahat = oo_.occbin.linear_smoother.etahat;
        end
        R.smoother = sm;
    end
    R.SmoothedVariables = oo_.SmoothedVariables;
    R.SmoothedShocks = oo_.SmoothedShocks;
    [dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    R.estim_param_names = bayestopt_.name;
    R.xparam_init = xparam_init;
    R.order_var = oo_.dr.order_var;
    R.restrict_var_list = oo_.dr.restrict_var_list;
    R.endo_names = M_.endo_names;
    pts = {xparam_init};
    rng(7);
    for k = 1:5
        pts{end+1} = xparam_init .* (1 + 0.1*randn(size(xparam_init)));
    end
    ivf = options_.occbin.likelihood.inversion_filter;
    R.points = {};
    for k = 1:numel(pts)
        x = pts{k};
        OCCBIN_PKF_DEBUG = [];
        tic;
        if ivf
            [fval, info, ~, ~, ~, ~, ~, ~, ~, atT, innov, rh] = occbin.IVF_posterior(x, dataset_, dataset_info, options_, M_, ...
                estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            lik = [];
        else
            [fval, info, ~, lik] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        end
        secs = toc;
        lp = priordens(x, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
        p = struct('x', x, 'fval', fval, 'info', info(1), 'log_prior', lp, 'lik', lik, 'seconds', secs);
        if ivf && info(1) == 0
            p.atT = atT; p.innov = innov;
            p.regime = arrayfun(@(x) double(x.regime), rh, 'UniformOutput', false);
            p.regimestart = arrayfun(@(x) double(x.regimestart), rh, 'UniformOutput', false);
        end
        if ~isempty(OCCBIN_PKF_DEBUG)
            p.filter_regime = arrayfun(@(x) double(x.regime), OCCBIN_PKF_DEBUG.regimes, 'UniformOutput', false);
            p.filter_regimestart = arrayfun(@(x) double(x.regimestart), OCCBIN_PKF_DEBUG.regimes, 'UniformOutput', false);
            p.filter_a0 = OCCBIN_PKF_DEBUG.a0;
            p.filter_lik = OCCBIN_PKF_DEBUG.lik;
            p.filter_t = OCCBIN_PKF_DEBUG.t;
        end
        R.points{end+1} = p;
        fprintf('%s point %d: fval %.12g info %d (%.2f s)\n', variants{v, 2}, k, fval, info(1), secs);
    end
    out.(variants{v, 2}) = R;
end
fid = fopen(fullfile(here, 'ref_occbin_est.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
delete('missing_observations_kalman_filter.m');
disp('wrote ref_occbin_est.json');
