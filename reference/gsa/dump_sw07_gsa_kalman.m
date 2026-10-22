% dump_sw07_gsa_kalman.m -- how well Dynare's own log posterior is determined at the stable
% SW07 prior draws of ref_gsa_sw07_stage21.json (tests/kalman_robust.rs).
%
%
% On the prior support some draws make the forecast-error covariance F badly conditioned, so
% the multivariate filter's value -- and whether it stops at a badly conditioned F
% (univariate fallback) or meets a negative det(F) (info 46) -- may be decided by rounding.
% dsge_likelihood is evaluated at each draw x and at three perturbations
% x .* (1 + 1e-14 randn): a draw is "determined" when the four values agree to 1e-6
% (relative), or all four fail.
%
% Run from a private copy of reference/sw07 (needs sw07/Output/sw07_mode.mat for the STAGE 9
% set-up, whose data and filter options are those of STAGE 21) with this file and
% ref_gsa_sw07_stage21.json next to sw07.mod:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_sw07_gsa_kalman.m')"
% Writes ref_gsa_sw07_kalman.json.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive nograph
cd(here);
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R0 = jsondecode(fileread('ref_gsa_sw07_stage21.json'));
ist = R0.istable;
rng(1);
n = numel(ist);
F = NaN(n, 4); INFO = NaN(n, 4);
for j = 1:n
    k = ist(j);
    x = [R0.lpmat0(k,:) R0.lpmat(k,:)]';
    for r = 1:4
        if r == 1, xr = x; else, xr = x .* (1 + 1e-14 * randn(size(x))); end
        [fval, info] = dsge_likelihood(xr, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        F(j, r) = fval; INFO(j, r) = info(1);
    end
end
fid = fopen('ref_gsa_sw07_kalman.json', 'w');
fprintf(fid, '%s', jsonencode(struct('fval', F, 'info', INFO), 'ConvertInfAndNaN', true));
fclose(fid);
