% dump_hess.m -- Dynare 7.2's Hessians at the mode of the mode_compute=5 runs of dump_runs.m,
% from a fresh hess_info (h1 = gradient_epsilon): mr_hessian on the univariate filter with
% newratflag 0 (outer product of the per-observable scores) and 1 (Hessian=2: numerical
% diagonal), as dynare_estimation_1.m computes them (ref_hess.json).
% Run from a PRIVATE COPY of this folder after dump_runs.m:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_hess.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
runs = jsondecode(fileread('ref_runs.json'));
dynare est_ad -DSTAGE=1 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = struct();
for key = {'mc5_hess0', 'mc5_hess2'}
    x = runs.(key{1}).xparam1; fval = runs.(key{1}).fval;
    n = numel(x);
    hess_info = struct('gstep', options_.gstep, 'htol', 1e-5, 'htol_rel', NaN, 'h1', options_.gradient_epsilon*ones(n,1), 'robust', false);
    o = options_; o.kalman_algo = 2;
    flag = double(strcmp(key{1}, 'mc5_hess2'));
    hh = reshape(mr_hessian(x, @dsge_likelihood, fval, flag, 1e-5, hess_info, [bounds.lb bounds.ub], bayestopt_.p2, 0, ...
        dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state), n, n);
    R.(key{1}) = struct('x', x, 'fval', fval, 'hh', hh);
end
fid = fopen('ref_hess.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
