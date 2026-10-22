% dump_reference.m -- run Dynare on the SW07 model (STAGE 9: mode, no MCMC)
% and write every intermediate the Rust port is checked against to ref.json.
%
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_reference.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);

mode = load('sw07/Output/sw07_mode.mat');
xm = mode.xparam1(:);

R = struct();
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.param_names = M_.param_names;
R.orig_endo_nbr = M_.orig_endo_nbr; R.lead_lag_incidence = M_.lead_lag_incidence;
R.aux_vars = M_.aux_vars; R.params_calib = M_.params; R.Sigma_e = M_.Sigma_e;
R.steady_state = oo_.steady_state;
R.estim_param_names = mode.parameter_names; R.xparam_mode = xm; R.xparam_init = xparam_init;
R.hh_mode = mode.hh;
R.bayestopt = struct('pshape', bayestopt_.pshape, 'p1', bayestopt_.p1, 'p2', bayestopt_.p2, ...
    'p3', bayestopt_.p3, 'p4', bayestopt_.p4, 'p5', bayestopt_.p5, 'p6', bayestopt_.p6, 'p7', bayestopt_.p7, ...
    'mf0', bayestopt_.mf0, 'mf1', bayestopt_.mf1, 'mfys', bayestopt_.mfys, ...
    'smoother_var_list', bayestopt_.smoother_var_list, 'smoother_mf', bayestopt_.smoother_mf);
R.bounds = struct('lb', bounds.lb, 'ub', bounds.ub);
R.options = struct('kalman_tol', options_.kalman_tol, 'diffuse_kalman_tol', options_.diffuse_kalman_tol, ...
    'riccati_tol', options_.riccati_tol, 'qz_criterium', options_.qz_criterium, 'qz_zero_threshold', options_.qz_zero_threshold, ...
    'lik_init', options_.lik_init, 'kalman_algo', options_.kalman_algo, 'presample', options_.presample, ...
    'first_obs', options_.first_obs, 'nobs', options_.nobs, 'lyapunov_complex_threshold', options_.lyapunov_complex_threshold, ...
    'use_univariate_filters_if_singularity_is_detected', options_.use_univariate_filters_if_singularity_is_detected);
R.data = dataset_.data;                      % nobs x nvobs, NaN = missing
R.varobs = options_.varobs;

% likelihood and prior at several points
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

% decision rule and state space at the mode
M2 = set_all_parameters(xm, estim_params_, M_);
[T, Rm, SS, info, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
Q = M2.Sigma_e;
[Pstar, Pinf] = compute_Pinf_Pstar(bayestopt_.mf1, T, Rm, Q, options_.qz_criterium, dr.restrict_columns);
R.mode_dr = struct('ys', dr.ys, 'ghx', dr.ghx, 'ghu', dr.ghu, 'order_var', dr.order_var, ...
    'inv_order_var', dr.inv_order_var, 'state_var', dr.state_var, 'npred', M_.npred, 'nboth', M_.nboth, ...
    'nstatic', M_.nstatic, 'nfwrd', M_.nfwrd, 'restrict_var_list', dr.restrict_var_list, ...
    'restrict_columns', dr.restrict_columns, 'eigval_abs', abs(dr.eigval));
ys = dr.ys; x0 = oo_.exo_steady_state';
[g1d] = feval([M_.fname '.dynamic_g1'], [ys; ys; ys], x0, M2.params, ys, ...
    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
R.mode_g1 = full(g1d);
g1s = feval([M_.fname '.static_g1'], ys, x0, M2.params, ...
    M_.static_g1_sparse_rowval, M_.static_g1_sparse_colval, M_.static_g1_sparse_colptr);
R.static_g1_mode = full(g1s);
R.mode_ss = struct('T', T, 'R', Rm, 'Q', Q, 'H', M2.H, 'Pstar', Pstar, 'Pinf', Pinf, 'SteadyState', SS, ...
    'params', M2.params);

% per-period likelihood from the diffuse filter + the rest
Z = bayestopt_.mf1; mm = length(T); pp = dataset_.vobs;
Zm = zeros(pp, mm); for i = 1:pp, Zm(i, Z(i)) = 1; end
Y = transpose(dataset_.data) - repmat(SS(bayestopt_.mfys), 1, dataset_.nobs);
[dLIK, dlik, a1, Pstar1] = missing_observations_kalman_filter_d(dataset_info.missing.aindex, ...
    dataset_info.missing.number_of_observations, dataset_info.missing.no_more_missing_observations, ...
    Y, 1, size(Y,2), T*zeros(mm,1), Pinf, Pstar, options_.kalman_tol, options_.diffuse_kalman_tol, ...
    options_.riccati_tol, options_.presample, T, Rm, Q, M2.H, Zm, mm, pp, length(Q));
if isfinite(dLIK)
    path = 'multivariate';
    [LIK, lik] = missing_observations_kalman_filter(dataset_info.missing.aindex, dataset_info.missing.no_more_missing_observations, ...
        Y, length(dlik)+1, size(Y,2), a1, Pstar1, options_.kalman_tol, options_.riccati_tol, ...
        options_.rescale_prediction_error_covariance, options_.presample, T, Q, Rm, M2.H, Zm, mm, pp, ...
        options_.varobs, options_.debug, 1, length(dlik), struct('status', false));
else
    % dsge_likelihood: singular diffuse filter -> univariate (kalman_algo 4), H = 0 -> H1 = zeros
    path = 'univariate';
    H1 = zeros(pp,1);
    [dLIK, dlik, a1, Pstar1] = univariate_kalman_filter_d(dataset_info.missing.aindex, Y, 1, size(Y,2), ...
        T*zeros(mm,1), Pinf, Pstar, options_.kalman_tol, options_.diffuse_kalman_tol, options_.presample, ...
        T, Rm, Q, H1, Zm, pp);
    [LIK, lik] = univariate_kalman_filter(dataset_info.missing.aindex, dataset_info.missing.no_more_missing_observations, ...
        Y, size(dlik,1)+1, size(Y,2), a1, Pstar1, options_.kalman_tol, options_.riccati_tol, options_.presample, ...
        T, Q, Rm, H1, Zm, mm, pp, 1, size(dlik,1), false, [], [], [], [], [], '');
end
R.mode_kalman_path = path;
R.mode_kalman = struct('Y', Y, 'dLIK', dLIK, 'dlik', dlik, 'LIK', LIK, 'lik', lik, 'a_after_diffuse', a1, ...
    'Pstar_after_diffuse', Pstar1, 'diffuse_periods', size(dlik,1), 'total', LIK+dLIK);

% smoother output from the STAGE 9 run (at the mode)
R.SmoothedVariables = oo_.SmoothedVariables;
R.SmoothedShocks = oo_.SmoothedShocks;
R.FilteredVariables = oo_.FilteredVariables;

% timing baseline: dsge_likelihood evaluations per second at the mode
nrep = 300; tic;
for k = 1:nrep
    dsge_likelihood(xm, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
end
R.timing_seconds_per_lik = toc/nrep;

fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('dump written; %.3f ms per likelihood\n', 1000*R.timing_seconds_per_lik);
