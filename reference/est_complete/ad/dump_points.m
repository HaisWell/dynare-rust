% dump_points.m -- Dynare reference values (DYNARE_VERSION, default 7.2) for analytic_derivation (est_ad.mod): at a few
% parameter vectors, dsge_likelihood's fval, DLIK and Hess for analytic_Hessian '', 'opg',
% 'full' and 'asymptotic' (ref_points_<stage>.json).
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_points.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
for stage = 1:5
cd(here);
eval(sprintf('dynare est_ad -DSTAGE=%d -DAD=1 nointeractive', stage));
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = struct();
R.names = bayestopt_.name; R.x_init = xparam_init;
pts = {xparam_init};
rng(11);
for k = 1:2
    pts{end+1} = xparam_init + 0.03*abs(xparam_init).*randn(size(xparam_init));
end
R.points = {};
for k = 1:numel(pts)
    x = pts{k};
    P = struct('x', x);
    o = options_; o.analytic_derivation = 0;
    P.fval_plain = dsge_likelihood(x, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    o.analytic_derivation = 1;
    for kind = {'', 'opg', 'full', 'asymptotic'}
        o.analytic_Hessian = kind{1};
        [fval, info, ~, DLIK, Hess] = dsge_likelihood(x, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        nm = kind{1}; if isempty(nm), nm = 'none'; end
        P.(nm) = struct('fval', fval, 'info', info(1), 'DLIK', DLIK(:), 'Hess', Hess);
    end
    R.points{end+1} = P;
end
fid = fopen(sprintf('ref_points_%d.json', stage), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('stage %d written\n', stage);
end
