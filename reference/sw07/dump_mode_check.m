% dump_mode_check.m -- reference values of mode_check.m (ref_mode_check.json).
%
% STAGE 9 run (mode file, no MCMC), then mode_check(@dsge_likelihood, ...) exactly as
% dynare_estimation_1.m calls it, with nograph; writes, per estimated parameter, the grid z
% and [-fval, -(fval + lnprior - lnprior(mode))] of mcheck.cross, plus the mode.
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_mode_check.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
options_.nograph = true;
options_.mode_check.status = true;
fval = dsge_likelihood(xparam1, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
    oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
tic;
mcheck = mode_check(@dsge_likelihood, xparam1, hh, options_, M_, estim_params_, bayestopt_, bounds, false, ...
    dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, ...
    oo_.exo_steady_state, oo_.exo_det_steady_state);
secs = toc;
names = fieldnames(mcheck.cross);
R = struct('xparam1', xparam1, 'fval', fval, 'seconds', secs, 'lb', bounds.lb, 'ub', bounds.ub);
R.names = names;
R.params = cell(numel(names), 1);
for k = 1:numel(names)
    c = mcheck.cross.(names{k});
    R.params{k} = struct('name', names{k}, 'mode', mcheck.emode.(names{k}), 'z', c(:,1), ...
        'log_post', c(:,2), 'log_lik', c(:,3));
end
fid = fopen('ref_mode_check.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('mode_check: %d parameters, %d points, %.1f s\n', numel(names), sum(cellfun(@(p) numel(p.z), R.params)), secs);
