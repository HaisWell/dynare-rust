% dump_skewed.m -- Dynare reference values (DYNARE_VERSION, default 7.2) for the skewed Kalman filter / smoother
% (kalman_algo = 5), sk_nk.mod. Run from a PRIVATE COPY of reference/est_complete/skewed
% (never inside a source checkout):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_skewed.m')"
% Writes ref_skewed_<stage>.json (objective at three points, smoother output or the error
% message) and ref_skewed_mvncdf.json (Dynare's mvncdf / csn_mean on fixed inputs).
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = pwd;
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info
if ~exist('STAGES', 'var')
    STAGES = 1:10;
end
for stage = STAGES
    cd(here);
    R = struct();
    try
        eval(sprintf('dynare sk_nk -DSTAGE=%d nolog nointeractive', stage));
    catch e
        R.error = e.message;
        fid = fopen(fullfile(here, sprintf('ref_skewed_%d.json', stage)), 'w');
        fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
        continue
    end
    if isfield(oo_, 'SmoothedVariables')
        R.SmoothedVariables = oo_.SmoothedVariables;
        R.SmoothedShocks = oo_.SmoothedShocks;
        if isfield(oo_, 'SmoothedMeasurementErrors')
            R.SmoothedMeasurementErrors = oo_.SmoothedMeasurementErrors;
        end
        if isfield(oo_, 'UpdatedVariables')
            R.UpdatedVariables = oo_.UpdatedVariables;
        end
    end
    options_.nobs = NaN;
    [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    R.xparam_init = xparam_init;
    R.nobs = dataset_.nobs;
    n = numel(xparam_init);
    pts = {xparam_init, xparam_init .* (1 + 0.03 * sin(1:n)'), xparam_init * 0.95};
    R.points = {};
    for k = 1:numel(pts)
        [fval, info, ~, DLIK] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        % a second evaluation (random QMC in mvncdf for dimensions >= 4)
        fval2 = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        R.points{end+1} = struct('x', pts{k}, 'fval', fval, 'fval2', fval2, 'info', info(1), 'DLIK', DLIK);
    end
    fid = fopen(fullfile(here, sprintf('ref_skewed_%d.json', stage)), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
end

% Dynare's mvncdf (missing/stats, missing/stats_matlab) and csn_mean on fixed inputs
cd(here);
C = struct();
C.bv = {};
S2 = {[1 0.5; 0.5 2], [1 -0.8; -0.8 1], [0.5 0.96; 0.96 2], [2 -0.97; -0.97 0.6], [1 0.1; 0.1 1]};
X2 = [0.3 -0.4; -1.2 0.7; 0.0 0.0; 2.5 -1.5; -3 4];
for k = 1:numel(S2)
    for j = 1:size(X2, 1)
        C.bv{end+1} = struct('x', X2(j,:), 'mu', [0.1 -0.2], 'Sigma', S2{k}, 'p', mvncdf(X2(j,:), [0.1 -0.2], S2{k}));
    end
end
C.tv = {};
S3 = {[1 0.3 0.2; 0.3 1 0.1; 0.2 0.1 1], [2 0.9 -0.5; 0.9 1 -0.3; -0.5 -0.3 1.5], [1 -0.6 0.4; -0.6 1 -0.95; 0.4 -0.95 1.2]};
X3 = [0.5 0.3 0.2; -1 0.4 1.3; 1.5 -0.5 -2; 0 0 0];
for k = 1:numel(S3)
    for j = 1:size(X3, 1)
        C.tv{end+1} = struct('x', X3(j,:), 'mu', [0.2 0 -0.1], 'Sigma', S3{k}, 'p', mvncdf(X3(j,:), [0.2 0 -0.1], S3{k}));
    end
end
C.qmc = {};
S5 = [2.0 0.5 0.2 0.1 0.3; 0.5 1.5 -0.3 0.4 0.1; 0.2 -0.3 1.8 0.2 -0.2; 0.1 0.4 0.2 1.2 0.05; 0.3 0.1 -0.2 0.05 1.0];
x5 = [0.4 0.6 -0.2 1.0 0.3];
ps = zeros(20, 1);
for r = 1:20
    ps(r) = mvncdf(x5, zeros(1,5), S5);
end
C.qmc{end+1} = struct('x', x5, 'Sigma', S5, 'p', ps);
% csn_mean, both CDF routines
C.csn = {};
mu = [2; -1; 0];
Sigma = [0.16 0.10 0.00; 0.10 0.64 -0.05; 0.00 -0.05 0.25];
Gs = {[2 0 0.7; 1 -0.3 0], [2 0 0.7; 1 -0.3 0; 0.5 0.5 -1], [2 0 0.7; 1 -0.3 0; 0.5 0.5 -1; -0.4 1 0.2]};
Ns = {[2; -0.5], [2; -0.5; 0.3], [2; -0.5; 0.3; -0.1]};
Ds = {[1 0.12; 0.12 1], [1 0.12 0.05; 0.12 1 -0.2; 0.05 -0.2 1.3], [1 0.12 0.05 0; 0.12 1 -0.2 0.1; 0.05 -0.2 1.3 0.3; 0 0.1 0.3 0.9]};
for k = 1:numel(Gs)
    C.csn{end+1} = struct('mu', mu, 'Sigma', Sigma, 'Gamma', Gs{k}, 'nu', Ns{k}, 'Delta', Ds{k}, ...
        'mean_me', csn_mean(mu, Sigma, Gs{k}, Ns{k}, Ds{k}, 'gaussian_log_mvncdf_mendell_elston'), ...
        'mean_mvncdf', csn_mean(mu, Sigma, Gs{k}, Ns{k}, Ds{k}, 'mvncdf'));
end
% csn_statespace_linear_transform: full rank (square / tall / wide) and rank deficient
C.lt = {};
Sb = [1 0.2 0.1 0; 0.2 0.8 0 0.1; 0.1 0 0.5 0.05; 0 0.1 0.05 0.7];
Gb = [0.5 -0.2 0 0.3; 0 1 0.4 -0.1];
nb = [0.1; -0.2];
Db = [1 0.1; 0.1 0.9];
mb = [0.1; 0.2; -0.3; 0.4];
Gm = {[0.9 0.1 0 0.2; 0 0.5 0.3 0; 0.1 0 0.7 0; 0 0 0.2 0.6], [0.9 0.1 0 0.2; 0 0.5 0.3 0], ...
      [0.9 0.1 0 0.2; 0 0.5 0.3 0; 0.9 0.1 0 0.2; 0 0 0.2 0.6], [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 1 1 0 0]};
Sbs = {Sb, Sb, Sb, blkdiag([1 0.2; 0.2 0.8], zeros(2))};
for k = 1:numel(Gm)
    [m_, S_, G_, n_, D_] = csn_statespace_linear_transform(Gm{k}, mb, Sbs{k}, Gb, nb, Db);
    C.lt{end+1} = struct('G', Gm{k}, 'mu', mb, 'Sigma', Sbs{k}, 'Gamma', Gb, 'nu', nb, 'Delta', Db, ...
        'mu_tr', m_, 'Sigma_tr', S_, 'Gamma_tr', G_, 'nu_tr', n_, 'Delta_tr', D_);
end
fid = fopen(fullfile(here, 'ref_skewed_mvncdf.json'), 'w');
fprintf(fid, '%s', jsonencode(C, 'ConvertInfAndNaN', true)); fclose(fid);
