% dump_point15.m -- Dynare internals of dsge_likelihood at points_wide.json point 16 (0-based 15):
% steady state, restricted T/R/Q, Pstar/Pinf, and the filters dsge_likelihood runs (no
% missing data, lik_init 3): kalman_filter_d -> (univariate_kalman_filter_d) -> kalman_filter
% -> (univariate_kalman_filter), with their outputs. Writes ref_point15.json.
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_point15.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end
addpath(fullfile(droot, 'matlab'));
cd(fileparts(mfilename('fullpath')));
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
bounds.lb(:) = -Inf; bounds.ub(:) = Inf;
P = jsondecode(fileread('points_wide.json'));
X = P.x; if size(X,2) ~= numel(bayestopt_.name), X = X'; end
k = 16;
x = X(k,:)';
[fval, info, ~, DLIK] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
    oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
M2 = set_all_parameters(x, estim_params_, M_);
[T, Rm, SS, info2, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
Q = M2.Sigma_e;
[Pstar, Pinf] = compute_Pinf_Pstar(bayestopt_.mf1, T, Rm, Q, options_.qz_criterium, dr.restrict_columns);
mm = length(T); pp = dataset_.vobs; rr = length(Q);
Z = zeros(pp, mm); for i = 1:pp, Z(i, bayestopt_.mf1(i)) = 1; end
Y = transpose(dataset_.data) - repmat(SS(bayestopt_.mfys), 1, dataset_.nobs);
kt = options_.kalman_tol; dkt = options_.diffuse_kalman_tol; rt = options_.riccati_tol; ps = options_.presample;
R = struct('k', k, 'x', x, 'fval', fval, 'info', info(1), 'DLIK', DLIK, 'steady_state', SS, 'T', T, 'R', Rm, 'Q', Q, ...
    'Pstar', Pstar, 'Pinf', Pinf, 'Y', Y);
% MATLAB's (unbalanced, unordered) real Schur form of the dynamic block, as compute_Pinf_Pstar sees it
indx = dr.restrict_columns;
[QT0, ST0] = schur(T(indx, indx));
R.schur_U = QT0; R.schur_T = ST0; R.schur_wr = real(ordeig(ST0)); R.schur_wi = imag(ordeig(ST0));
[dLIK, dlik, a1, Pstar1] = kalman_filter_d(Y, 1, size(Y,2), zeros(mm,1), Pinf, Pstar, kt, dkt, ps, T, Rm, Q, 0, Z, pp);
R.mv_d = struct('dLIK', dLIK, 'dlik', dlik, 'n', length(dlik));
if isinf(dLIK)
    [dLIK, dlik, a1, Pstar1] = univariate_kalman_filter_d(dataset_info.missing.aindex, Y, 1, size(Y,2), zeros(mm,1), ...
        Pinf, Pstar, kt, dkt, ps, T, Rm, Q, zeros(pp,1), Z, pp);
    R.uv_d = struct('dLIK', dLIK, 'dlik', dlik, 'n', size(dlik,1));
end
d = size(dlik,1);
R.a_after_diffuse = a1; R.Pstar_after_diffuse = Pstar1; R.diffuse_periods = d;
[LIK, lik] = kalman_filter(Y, d+1, size(Y,2), a1, Pstar1, kt, rt, options_.rescale_prediction_error_covariance, ps, ...
    T, Q, Rm, 0, Z, mm, pp, rr, 1, d, options_.varobs, 0, false);
R.mv = struct('LIK', LIK, 'lik', lik);
% the same filter without the steady-state shortcut (riccati_tol = 0)
[LIK0, lik0] = kalman_filter(Y, d+1, size(Y,2), a1, Pstar1, kt, 0, options_.rescale_prediction_error_covariance, ps, ...
    T, Q, Rm, 0, Z, mm, pp, rr, 1, d, options_.varobs, 0, false);
R.mv_noss = struct('LIK', LIK0, 'lik', lik0);
[LIKu, liku] = univariate_kalman_filter(dataset_info.missing.aindex, dataset_info.missing.no_more_missing_observations, ...
    Y, d+1, size(Y,2), a1, Pstar1, kt, rt, ps, T, Q, Rm, zeros(pp,1), Z, mm, pp, 1, d, false, [], [], [], [], [], '');
R.uv = struct('LIK', LIKu, 'lik', liku);
fid = fopen('ref_point15.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('fval %.10f  mv_d %g  d %d  LIK(mv) %.10f  LIK(mv, no ss) %.10f  LIK(uv) %.10f\n', fval, R.mv_d.dLIK, d, LIK, LIK0, LIKu);
