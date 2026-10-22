% dump_q3.m -- Dynare 7.2 reference values for (estimation options), q3_nk.mod.
% Run from a PRIVATE COPY of reference/q3 (never inside a source checkout):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q3.m')"
% Writes ref_q3_<stage>.json (objective at three points, smoother output, DSGE-VAR posterior
% objects) and ref_q3_convergence.json (raftery_lewis / geweke_moments on a fixed sequence).
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end;
addpath(fullfile(droot, 'matlab'));
here = pwd;
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info
if ~exist('STAGES', 'var')
    STAGES = [1:9 10 11];
end
for stage = STAGES
    cd(here);
    if stage == 11
        % Bayesian IRF matching with an MCMC (nk_irf_mcmc.mod, CASE 2)
        dynare nk_irf_mcmc -DCASE=2 nolog nointeractive
        R = struct('posterior_mean', oo_.mom.posterior_mean, 'mode', oo_.mom.posterior_mode, 'mhm', oo_.mom.MarginalDensity.ModifiedHarmonicMean);
        fid = fopen(fullfile(here, 'ref_q3_11.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
        continue
    end
    eval(sprintf('dynare q3_nk -DSTAGE=%d nolog nointeractive', stage));
    if stage == 10
        R = struct('posterior_mean', oo_.posterior_mean, 'posterior_mode', oo_.posterior_mode, 'mhm', oo_.MarginalDensity.ModifiedHarmonicMean, ...
                   'irf_bvardsge_mean', oo_.PosteriorIRF.bvardsge.Mean, 'irf_dsge_mean', oo_.PosteriorIRF.dsge.Mean);
        fid = fopen(fullfile(here, 'ref_q3_10.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
        continue
    end
    R = struct();
    if isfield(oo_, 'FilteredVariablesKStepAhead') && stage >= 12
        R.KStepAhead = oo_.FilteredVariablesKStepAhead;
        if isfield(oo_, 'FilteredVariablesKStepAheadVariances')
            R.KStepAheadVariances = oo_.FilteredVariablesKStepAheadVariances;
        end
        if isfield(oo_, 'FilteredVariablesShockDecomposition')
            R.ShockDecomposition = oo_.FilteredVariablesShockDecomposition;
        end
    end
    if isfield(oo_, 'SmoothedVariables')
        R.SmoothedVariables = oo_.SmoothedVariables;
        R.SmoothedShocks = oo_.SmoothedShocks;
    end
    options_.nobs = NaN; % reset by the estimation's own dynare_estimation_init
    [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    R.xparam_init = xparam_init;
    R.data = dataset_.data;
    R.nobs = dataset_.nobs;
    n = numel(xparam_init);
    pts = {xparam_init, xparam_init .* (1 + 0.03 * sin(1:n)'), xparam_init * 0.95};
    if options_.dsge_var
        if options_.noconstant
            dataset_info = var_sample_moments(options_.dsge_varlag, -1, dataset_, dataset_info);
        else
            dataset_info = var_sample_moments(options_.dsge_varlag, 0, dataset_, dataset_info);
        end
        R.mYY = dataset_info.mYY; R.mYX = dataset_info.mYX; R.mXX = dataset_info.mXX;
        fun = @dsge_var_likelihood;
    elseif options_.conditional_likelihood.status
        fun = @dsge_conditional_likelihood_1;
    else
        fun = @dsge_likelihood;
    end
    R.points = {};
    for k = 1:numel(pts)
        if options_.dsge_var
            [fval, info, ~, ~, ~, ~, ~, PHI, SIGMAu, iXX, prior] = fun(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            R.points{end+1} = struct('x', pts{k}, 'fval', fval, 'info', info(1), 'PHI_tilde', PHI, 'SIGMA_u_tilde', SIGMAu, 'iXX', iXX, 'prior', prior);
        else
            [fval, info] = fun(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            R.points{end+1} = struct('x', pts{k}, 'fval', fval, 'info', info(1));
        end
    end
    fid = fopen(fullfile(here, sprintf('ref_q3_%d.json', stage)), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
end

% convergence diagnostics on a fixed sequence
C = struct();
N = 5000;
x = zeros(N, 2);
for t = 2:N
    x(t, 1) = 0.8 * x(t-1, 1) + sin(1.7 * t) + 0.5 * cos(0.3 * t^1.1);
    x(t, 2) = 0.3 * x(t-1, 2) + sin(2.9 * t^1.05);
end
C.x = x;
C.rl = raftery_lewis(x, 0.025, 0.005, 0.95);
C.rl2 = raftery_lewis(x, 0.1, 0.02, 0.9);
opt.convergence.geweke.taper_steps = [2 5 10];
C.gm = geweke_moments(x(:, 1), opt);
C.q = [quantile(x(:, 1), 0.025), quantile(x(:, 2), 0.3)];
fid = fopen(fullfile(here, 'ref_q3_convergence.json'), 'w');
fprintf(fid, '%s', jsonencode(C, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dumps written');
