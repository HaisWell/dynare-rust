% dump_modes.m -- analytic_derivation_mode 1, -1, -2 (est_ad.mod stage 6, no measurement error;
% with estimated measurement errors mode -1 fails in Dynare 7.1: reshape error): dsge_likelihood's
% DLIK and the 'full' Hess at the initial vector (ref_modes.json).
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_modes.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare est_ad -DSTAGE=6 -DAD=1 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, x, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = struct('x', x);
for mode = [1 -1 -2]
    o = options_; o.analytic_derivation = 1; o.analytic_derivation_mode = mode;
    o.analytic_Hessian = '';
    [fval, info, ~, DLIK] = dsge_likelihood(x, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    o.analytic_Hessian = 'full';
    [~, ~, ~, ~, Hess] = dsge_likelihood(x, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    R.(sprintf('mode_%s', strrep(num2str(mode), '-', 'm'))) = struct('fval', fval, 'DLIK', DLIK(:), 'Hess', Hess);
end
fid = fopen('ref_modes.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
