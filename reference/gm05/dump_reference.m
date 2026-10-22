% dump_reference.m -- Dynare 7.2 reference values for model_gm05.mod (STAGE 1 setup:
% estimated_params_init(use_calibration), first_obs=1, diffuse_filter), written to ref.json.
%
% Run from a PRIVATE COPY of this folder (Dynare writes +model_gm05/ and model_gm05/ here):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_reference.m')"
%
% The estimation call of STAGE 1 is patched to mode_compute=0 (no optimisation), so
% dynare_estimation_1 only runs the smoother at the initial values. Points evaluated with
% dsge_likelihood: the initial vector, the mode file's xparam1 mapped by name onto the current
% estimated_params (missing entries keep their initial value, as check_mode_file.m does when
% mode_compute > 0), and four perturbations of it.
fname = 'model_gm05';
modefile = 'model_gm05/Output/model_gm05_mode.mat';
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
src = fileread([fname '.mod']);
src = strrep(src, 'mode_compute=4, mh_replic=0,', 'mode_compute=0, mh_replic=0, smoother,');
fid = fopen([fname '_dump.mod'], 'w'); fprintf(fid, '%s', src); fclose(fid);
eval(['dynare ' fname '_dump -DSTAGE=1 nointeractive']);
global M_ options_ oo_ estim_params_ bayestopt_
smoothed_init = oo_.SmoothedVariables; smoothed_shocks_init = oo_.SmoothedShocks;
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);

mode = load(modefile);
xm = xparam_init;
for i = 1:numel(xm)
    id = find(strcmp(mode.parameter_names, bayestopt_.name{i}));
    if ~isempty(id), xm(i) = mode.xparam1(id); end
end

R = struct();
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.param_names = M_.param_names;
R.lead_lag_incidence = M_.lead_lag_incidence;
R.params_calib = M_.params; R.Sigma_e = M_.Sigma_e; R.H = M_.H;
R.steady_state = oo_.steady_state;
R.estim_param_names = bayestopt_.name; R.xparam_mode = xm; R.xparam_init = xparam_init;
R.bayestopt = struct('pshape', bayestopt_.pshape, 'p1', bayestopt_.p1, 'p2', bayestopt_.p2, ...
    'p3', bayestopt_.p3, 'p4', bayestopt_.p4, 'p6', bayestopt_.p6, 'p7', bayestopt_.p7, ...
    'mf1', bayestopt_.mf1, 'mfys', bayestopt_.mfys);
R.bounds = struct('lb', bounds.lb, 'ub', bounds.ub);
R.options = struct('first_obs', options_.first_obs, 'nobs', options_.nobs, 'presample', options_.presample, ...
    'lik_init', options_.lik_init, 'kalman_algo', options_.kalman_algo);
R.data = dataset_.data;
R.varobs = options_.varobs;

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
    R.points{end+1} = struct('x', x, 'minus_loglik', fval, 'info', info(1), 'log_prior', lp);
end

M2 = set_all_parameters(xm, estim_params_, M_);
[T, Rm, SS, info, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
R.mode_dr = struct('ys', dr.ys, 'ghx', dr.ghx, 'ghu', dr.ghu, 'order_var', dr.order_var, ...
    'state_var', dr.state_var, 'restrict_var_list', dr.restrict_var_list, 'restrict_columns', dr.restrict_columns);
R.mode_ss = struct('T', T, 'R', Rm, 'Q', M2.Sigma_e, 'H', M2.H, 'SteadyState', SS, 'params', M2.params);

% smoother at xm
[atT, innov, measurement_error, updated_variables, ys] = DsgeSmoother(xm, dataset_.nobs, transpose(dataset_.data), ...
    dataset_info.missing.aindex, dataset_info.missing.state, M_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, ...
    oo_.exo_det_steady_state, options_, bayestopt_, estim_params_);
R.SmoothedVariables_init = smoothed_init; R.SmoothedShocks_init = smoothed_shocks_init;
R.smoother_atT = atT;            % restricted states (dr order of restrict_var_list) x nobs, demeaned
R.smoother_innov = innov;
R.smoother_measurement_error = measurement_error;

nrep = 100; tic;
for k = 1:nrep
    dsge_likelihood(xm, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
end
R.timing_seconds_per_lik = toc/nrep;

fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('dump written; %.3f ms per likelihood\n', 1000*R.timing_seconds_per_lik);
