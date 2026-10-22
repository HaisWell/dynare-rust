% dump_reference.m -- Dynare (DYNARE_VERSION, default 7.2) maximum-likelihood estimation of model_ml.mod
% (mode_compute=4, no priors), written to ref.json. Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_reference.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
cd(fileparts(mfilename('fullpath')));
dynare model_ml nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
mode = load('model_ml/Output/model_ml_mode.mat');
R = struct();
R.parameter_names = mode.parameter_names; R.xparam_mode = mode.xparam1; R.fval_mode = mode.fval;
R.hh = mode.hh;
R.mode = oo_.mle_mode; R.std_at_mode = oo_.mle_std_at_mode;
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R.xparam_init = xparam_init; R.bounds = struct('lb', bounds.lb, 'ub', bounds.ub);
pts = {mode.xparam1, xparam_init, [0.5; -0.1; 0.2; 1.2; 0.4; 0.05]};
R.points = {};
for k = 1:numel(pts)
    [fval, info] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    R.points{end+1} = struct('x', pts{k}, 'minus_loglik', fval, 'info', info(1));
end
fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dump written');
