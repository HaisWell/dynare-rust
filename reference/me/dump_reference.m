% dump_reference.m -- Dynare reference values (DYNARE_VERSION, default 7.2) for sw07_me.mod, the SW07 model (../sw07/sw07.mod) with
% measurement errors (STAGE 10: diagonal H; STAGE 11: correlated measurement errors),
% written to ref_<stage>.json.
%
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_reference.m')"
%
% The estimation runs with mode_compute=0 and no mode file (smoother at the initial values).
% Points: the SW07 mode (sw07/Output/sw07_mode.mat mapped by name; the measurement
% errors keep their initial values), the initial vector and four perturbations of the first.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
for stage = [10 11]
cd(here);
eval(sprintf('dynare sw07_me -DSTAGE=%d nointeractive', stage));
global M_ options_ oo_ estim_params_ bayestopt_
R = struct();
R.SmoothedVariables_init = oo_.SmoothedVariables; R.SmoothedShocks_init = oo_.SmoothedShocks;
R.SmoothedMeasurementErrors_init = oo_.SmoothedMeasurementErrors;
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
mode = load('sw07/Output/sw07_mode.mat');
xm = xparam_init;
for i = 1:numel(xm)
    id = find(strcmp(mode.parameter_names, bayestopt_.name{i}));
    if ~isempty(id), xm(i) = mode.xparam1(id); end
end
R.H_calib = M_.H;
R.estim_param_names = bayestopt_.name; R.xparam_mode = xm; R.xparam_init = xparam_init;
R.bounds = struct('lb', bounds.lb, 'ub', bounds.ub);
pts = {xm, xparam_init};
rng(42);
for k = 1:4
    pts{end+1} = xm + 0.02*abs(xm).*randn(size(xm));
end
R.points = {};
for k = 1:numel(pts)
    x = pts{k};
    [fval, info] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    lp = priordens(x, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
    M2 = set_all_parameters(x, estim_params_, M_);
    R.points{end+1} = struct('x', x, 'minus_loglik', fval, 'info', info(1), 'log_prior', lp, 'H', M2.H);
end
nrep = 50; tic;
for k = 1:nrep
    dsge_likelihood(xm, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
end
R.timing_seconds_per_lik = toc/nrep;
fid = fopen(sprintf('ref_%d.json', stage), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('stage %d dump written; %.3f ms per likelihood\n', stage, 1000*R.timing_seconds_per_lik);
end
