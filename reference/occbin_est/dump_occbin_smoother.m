% dump_occbin_smoother.m -- Dynare 7.2 reference values for the OccBin smoother
% (occbin.DSGE_smoother: realtime piecewise Kalman smoother, regime iterations, conditional
% smoother) of nk_est.mod at the parameter points of ref_occbin_est.json, written to
% ref_occbin_smoother.json. Run from a PRIVATE COPY of this folder after dump_occbin_est.m:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_smoother.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
ref = jsondecode(fileread('ref_occbin_est.json'));
variants = { '', 'pkf'; '-DIVF', 'ivf' };
out = struct();
for v = 1:size(variants, 1)
    clear functions
    eval(sprintf('dynare nk_est %s -DNOEST noclearall nograph nolog', variants{v, 1}));
    global M_ options_ oo_ estim_params_ bayestopt_
    options_.smoother = true;
    options_.mh_replic = 0;
    options_.datafile = 'nk_est_data.csv';
    options_.nograph = true;
    options_.occbin.smoother.plot = false;
    [dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    pts = ref.(variants{v, 2}).points;
    R = struct();
    R.points = {};
    for k = 1:numel(pts)
        if iscell(pts), p = pts{k}; else, p = pts(k); end
        x = p.x;
        oo1 = oo_;
        s = struct();
        if options_.occbin.smoother.inversion_filter
            [~, info, ~, ~, ~, ~, ~, ~, ~, atT, innov, rh] = occbin.IVF_posterior(x, dataset_, dataset_info, options_, M_, ...
                estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            s.alphahat = atT; s.etahat = innov; s.info = info(1);
            s.regime = arrayfun(@(x) double(x.regime), rh, 'UniformOutput', false);
            s.regimestart = arrayfun(@(x) double(x.regimestart), rh, 'UniformOutput', false);
        else
            tic;
            [atT, innov, ~, updated, ys, ~, ~, ~, ~, ~, ~, ~, ~, ~, oo1, ~, a0T] = occbin.DSGE_smoother(x, dataset_.nobs, transpose(dataset_.data), ...
                dataset_info.missing.aindex, dataset_info.missing.state, M_, oo_, options_, bayestopt_, estim_params_, dataset_, dataset_info);
            s.seconds = toc;
            s.alphahat = atT; s.etahat = innov; s.alphahat0 = a0T; s.updated = updated;
            sm = oo1.occbin.smoother;
            s.error_flag = sm.error_flag;
            if isfield(sm, 'warning_flag'), s.warning_flag = sm.warning_flag; end
            s.regime = arrayfun(@(x) double(x.regime), sm.regime_history, 'UniformOutput', false);
            s.regimestart = arrayfun(@(x) double(x.regimestart), sm.regime_history, 'UniformOutput', false);
            s.realtime_regime = arrayfun(@(x) double(x.regime), sm.realtime_regime_history, 'UniformOutput', false);
            s.realtime_regimestart = arrayfun(@(x) double(x.regimestart), sm.realtime_regime_history, 'UniformOutput', false);
            if isfield(sm, 'regime_history_iter'), s.iterations = size(sm.regime_history_iter, 1); end
            if isfield(oo1.occbin, 'linear_smoother')
                s.linear_alphahat = oo1.occbin.linear_smoother.alphahat;
                s.linear_etahat = oo1.occbin.linear_smoother.etahat;
            end
        end
        R.points{end+1} = s;
        fprintf('%s point %d done\n', variants{v, 2}, k);
    end
    out.(variants{v, 2}) = R;
end
fid = fopen(fullfile(here, 'ref_occbin_smoother.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin_smoother.json');
