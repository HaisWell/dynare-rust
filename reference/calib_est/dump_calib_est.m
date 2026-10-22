% dump_calib_est.m -- Dynare references (DYNARE_VERSION, default 7.2) for irf_calibration / moment_calibration applied
% inside the estimation and the other consumers of options_.endogenous_prior_restrictions
% (nk_ce.mod, CASE 1..5): ref_calib_est.json and the logs calib_est_c<k>_matlab.log.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_calib_est.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
global M_ options_ oo_ estim_params_ bayestopt_
out = struct();

% ---- CASE 1: mode + MCMC; dsge_likelihood at points inside / outside the restrictions ----
dynare nk_ce -DCASE=1 nointeractive noclearall
copyfile('nk_ce.log', 'calib_est_c1_matlab.log');
xm = oo_.posterior.optimization.mode;
out.c1.mode = xm;

out.c1.variance = oo_.posterior.optimization.Variance;
out.c1.posterior_mean = [cellfun(@(f) oo_.posterior_mean.shocks_std.(f), fieldnames(oo_.posterior_mean.shocks_std)); cellfun(@(f) oo_.posterior_mean.parameters.(f), fieldnames(oo_.posterior_mean.parameters))];
out.c1.irf = options_.endogenous_prior_restrictions.irf;
out.c1.moment = options_.endogenous_prior_restrictions.moment;
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
% every MCMC draw satisfies the restrictions
nviol = 0; ndraws = 0;
for b = 1:2
    S = load(fullfile('nk_ce', 'metropolis', sprintf('nk_ce_mh1_blck%d.mat', b)));
    for k = 1:size(S.x2, 1)
        Mk = set_all_parameters(S.x2(k, :)', estim_params_, M_);
        [T, R, ~, info, drk] = dynare_resolve(Mk, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
        inf2 = endogenous_prior_restrictions(T, R, Mk, options_, drk, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        nviol = nviol + (inf2(1) ~= 0);
        ndraws = ndraws + 1;
    end
end
out.c1.mcmc_draws = ndraws;
out.c1.mcmc_violations = nviol;
sd = sqrt(diag(out.c1.variance));
pts = {xm};
f = ones(size(xm)); f(1) = 1.1; pts{end+1} = xm .* f;           % y(1) to e_g above 1.6
f = ones(size(xm)); f(2) = 1.5; pts{end+1} = xm .* f;           % pi(1) violated, pi(2) holds
f = ones(size(xm)); f(2) = 3.0; pts{end+1} = xm .* f;           % pi(1:2) both violated
f = ones(size(xm)); f(1) = 1.3; f(2) = 3.0; pts{end+1} = xm .* f; % irf restrictions summed
f = ones(size(xm)); f(9) = 0.6; pts{end+1} = xm .* f;           % rho_g low: y autocorrelation < 0.5
rng(7);
for k = 1:40
    pts{end+1} = xm + 3 * sd .* randn(size(xm));
end
out.c1.points = {};
for k = 1:numel(pts)
    x = pts{k};
    [fval, info] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    P = struct('x', x, 'fval', fval, 'info', info(1), 'info4', NaN);
    if numel(info) >= 4
        P.info4 = info(4);
    end
    P.restr_info = [0 0];
    if all(x >= bounds.lb) && all(x <= bounds.ub)
        Mk = set_all_parameters(x, estim_params_, M_);
        [T, R, ~, inf1, drk] = dynare_resolve(Mk, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
        if ~inf1(1)
            [ri, ii, im, di, dm] = endogenous_prior_restrictions(T, R, Mk, options_, drk, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            P.restr_info = ri; P.info_irf = ii; P.info_moment = im; P.data_irf = di; P.data_moment = dm;
        end
    end
    out.c1.points{end+1} = P;
end
clear oo_ M_ options_ estim_params_ bayestopt_ dataset_ dataset_info

% ---- CASE 2: a moment restriction binding at the unrestricted mode ----
try
    dynare nk_ce -DCASE=2 nointeractive noclearall
    out.c2.mode = oo_.posterior.optimization.mode;
    out.c2.variance = oo_.posterior.optimization.Variance;
    out.c2.error = '';
catch ME
    out.c2.error = ME.message;
end
copyfile('nk_ce.log', 'calib_est_c2_matlab.log');
clear oo_ M_ options_ estim_params_ bayestopt_

% ---- CASE 3: prior simulate ----
dynare nk_ce -DCASE=3 nointeractive noclearall
copyfile('nk_ce.log', 'calib_est_c3_matlab.log');
clear oo_ M_ options_ estim_params_ bayestopt_

% ---- CASE 4: prior moments, restriction violated at the prior mode ----
dynare nk_ce -DCASE=4 nointeractive noclearall
copyfile('nk_ce.log', 'calib_est_c4_matlab.log');
clear oo_ M_ options_ estim_params_ bayestopt_

% ---- CASE 5: sensitivity (stability mapping with the restrictions) ----
dynare nk_ce -DCASE=5 nointeractive noclearall
copyfile('nk_ce.log', 'calib_est_c5_matlab.log');
P = load(fullfile('nk_ce', 'gsa', 'nk_ce_prior.mat'));
out.c5.lpmat = P.lpmat; out.c5.lpmat0 = P.lpmat0;
out.c5.infox = P.infox; out.c5.istable = P.istable; out.c5.irestriction = P.irestriction;
out.c5.iunstable = P.iunstable; out.c5.ixun = P.ixun; out.c5.iindeterm = P.iindeterm; out.c5.iwrong = P.iwrong;

fid = fopen(fullfile(here, 'ref_calib_est.json'), 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
disp('ref_calib_est.json written');
