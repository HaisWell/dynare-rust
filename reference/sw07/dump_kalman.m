% dump_kalman.m -- reference values for the Rust `kalman` module (ref_kalman.json).
%
% Reuses dump_reference.m's setup (STAGE 9 run, dynare_estimation_init, mode file) and,
% for several parameter points, dumps the restricted state space (T, R, Q, Pstar, Pinf and
% the lik_init=1 Lyapunov Pstar) plus dsge_likelihood's value and per-period likelihood for
% a grid of filter configurations:
%   lik_init 1/3  x  kalman_algo 0/1/2/3/4  x  data with/without missing values  x  H = 0 / diagonal.
% It also dumps DsgeSmoother's raw outputs (alphahat, etahat, epsilonhat, ahat, aK(1,:,:),
% some state_uncertainty slices) on the full (non-restricted, smoother_var_list) state space.
%
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_kalman.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);

R0 = jsondecode(fileread('ref.json'));
pts = R0.points;
if iscell(pts), pts = [pts{:}]; end

pp = dataset_.vobs; nobs = dataset_.nobs;
data_full = dataset_.data;                       % nobs x pp
% Missing-data variant: a few isolated holes, a fully missing period and a block.
data_miss = data_full;
data_miss(1:10, 3) = NaN;
data_miss(20, 5) = NaN;
data_miss(30, :) = NaN;
data_miss(40:45, 7) = NaN;
data_miss(12, [1 2 9]) = NaN;
% Diagonal measurement error: 5% of each series' variance.
Hdiag = diag(0.05*var(data_full, 0, 1));

datasets = {data_full, data_miss};
Hs = {0, Hdiag};

R = struct();
R.Hdiag = diag(Hdiag);
R.data_miss = data_miss;
R.mf1 = bayestopt_.mf1;
R.mfys = bayestopt_.mfys;
R.restrict_columns = oo_.dr.restrict_columns;
R.options = struct('kalman_tol', options_.kalman_tol, 'diffuse_kalman_tol', options_.diffuse_kalman_tol, ...
    'riccati_tol', options_.riccati_tol, 'presample', options_.presample, 'qz_criterium', options_.qz_criterium, ...
    'lyapunov_complex_threshold', options_.lyapunov_complex_threshold, ...
    'use_univariate_filters_if_singularity_is_detected', options_.use_univariate_filters_if_singularity_is_detected);

configs = [ ...   % lik_init kalman_algo missing H
    3 0 0 0; 3 4 0 0; 1 0 0 0; 1 2 0 0; ...
    3 0 1 0; 3 4 1 0; 1 0 1 0; 1 2 1 0; ...
    3 0 0 1; 3 4 0 1; 1 0 0 1; 1 2 0 1; ...
    3 0 1 1; 3 4 1 1; 1 0 1 1; 1 2 1 1; 3 3 1 1; 1 1 1 1];

R.points = {};
for k = [1 3 6]
    x = pts(k).x(:);
    M2 = set_all_parameters(x, estim_params_, M_);
    [T, Rm, SS, info, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
    Q = M2.Sigma_e;
    [Pstar, Pinf] = compute_Pinf_Pstar(bayestopt_.mf1, T, Rm, Q, options_.qz_criterium, dr.restrict_columns);
    Plyap = lyapunov_solver(T, Rm, Q, options_);
    P = struct('k', k, 'x', x, 'T', T, 'R', Rm, 'Q', Q, 'Pstar', Pstar, 'Pinf', Pinf, 'Plyap', Plyap, ...
        'constant', SS(bayestopt_.mfys), 'info', info(1));
    P.runs = {};
    for c = 1:size(configs, 1)
        cfg = configs(c, :);
        opts = options_;
        opts.lik_init = cfg(1);
        opts.kalman_algo = cfg(2);
        ds = struct('data', datasets{cfg(3)+1}, 'vobs', pp, 'nobs', nobs);
        dsi = dataset_info;
        [ai, nn, s] = describe_missing_data(ds.data);
        dsi.missing.aindex = ai;
        dsi.missing.number_of_observations = nn;
        dsi.missing.no_more_missing_observations = s;
        dsi.missing.state = any(isnan(ds.data(:)));
        Mk = M_;
        Mk.H = Hs{cfg(4)+1};
        try
            [fval, info, ~, DLIK] = dsge_likelihood(x, ds, dsi, opts, Mk, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            lnprior = -DLIK(1);
            lik = DLIK(2:end);
            ncol = numel(lik)/(nobs - opts.presample);
            lik = reshape(lik, nobs - opts.presample, ncol);
            err = '';
        catch ME
            fval = NaN; info = -1; lnprior = NaN; lik = []; ncol = 0; err = ME.message;
        end
        P.runs{end+1} = struct('lik_init', cfg(1), 'kalman_algo', cfg(2), 'missing', cfg(3), 'H', cfg(4), ...
            'fval', fval, 'info', info(1), 'lnprior', lnprior, 'LIK', fval + lnprior, 'lik', lik, ...
            'lik_cols', ncol, 'no_more_missing_observations', s, 'error', err);
        fprintf('point %d cfg [%d %d %d %d]: fval %.10f info %d cols %d %s\n', k, cfg, fval, info(1), ncol, err);
    end
    R.points{end+1} = P;
end

% ---- smoother (DsgeSmoother, smoother_redux = false) ----
mode = load('sw07/Output/sw07_mode.mat');
xm = mode.xparam1(:);
sm_configs = [ ...  % lik_init kalman_algo missing H
    3 0 0 0; 3 0 1 1; 1 0 1 1; 3 4 1 1; 1 2 0 0];
check_periods = [1 91];
R.smoother = {};
R.smoother_system = struct();
for c = 1:size(sm_configs, 1)
    cfg = sm_configs(c, :);
    opts = options_;
    opts.lik_init = cfg(1);
    opts.kalman_algo = cfg(2);
    opts.smoothed_state_uncertainty = true;
    data = datasets{cfg(3)+1};
    [ai, nn, s] = describe_missing_data(data);
    Mk = M_;
    Mk.H = Hs{cfg(4)+1};
    [alphahat, etahat, epsilonhat, ahat, SteadyState, trend_coeff, aK, Ts, Rs, Ps, PKs, decomp, trend_addition, ...
        state_uncertainty, drs, mfs, alphahat0, state_uncertainty0, d, info] = ...
        DsgeSmoother(xm, nobs, transpose(data), ai, any(isnan(data(:))), Mk, oo_.dr, oo_.steady_state, ...
        oo_.exo_steady_state, oo_.exo_det_steady_state, opts, bayestopt_, estim_params_);
    if c == 1
        % same parameters for every run: the smoother's (full, smoother_var_list) state space once
        Ms = set_all_parameters(xm, estim_params_, M_);
        Qs = Ms.Sigma_e;
        [Pstar_s, Pinf_s] = compute_Pinf_Pstar(mfs, Ts, Rs, Qs, options_.qz_criterium);
        R.smoother_system = struct('T', Ts, 'R', Rs, 'Q', Qs, 'mf', mfs, 'Pstar', Pstar_s, 'Pinf', Pinf_s, ...
            'Plyap', lyapunov_solver(Ts, Rs, Qs, options_), 'constant', SteadyState(bayestopt_.mfys), ...
            'order_var', drs.order_var, 'ys', SteadyState);
    end
    Vdiag = zeros(size(state_uncertainty, 1), nobs);
    for tt = 1:nobs, Vdiag(:, tt) = diag(state_uncertainty(:, :, tt)); end
    S = struct('lik_init', cfg(1), 'kalman_algo', cfg(2), 'missing', cfg(3), 'H', cfg(4), ...
        'd', d, 'info', info(1), ...
        'alphahat', alphahat, 'etahat', etahat, 'epsilonhat', epsilonhat, 'ahat', ahat, ...
        'aK1', squeeze(aK(1, :, :)), 'alphahat0', alphahat0, 'check_periods', check_periods, ...
        'V', state_uncertainty(:, :, check_periods), 'Vdiag', Vdiag);
    R.smoother{end+1} = S;
    fprintf('smoother cfg [%d %d %d %d]: d %d info %d\n', cfg, d, info(1));
end

fid = fopen('ref_kalman.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_kalman.json written\n');
