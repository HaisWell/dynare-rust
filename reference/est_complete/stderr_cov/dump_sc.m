% Run from a PRIVATE COPY of this folder: matlab -batch "run('dump_sc.m')"  -> ref_sc.json
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath')); cd(here);
R = struct();
for STAGE = 1:4
    clear global
    global M_ options_ oo_ estim_params_ bayestopt_
    eval(sprintf('dynare sc -DSTAGE=%d nointeractive', STAGE));
    [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    f = ones(size(xparam_init)); f(1:3) = [1.02; 0.97; 1.1];
    pts = {xparam_init, xparam_init .* f, xparam_init * 0.95};
    S = struct('xparam_init', xparam_init, 'points', {{}});
    for k = 1:numel(pts)
        [fval, info] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        S.points{end+1} = struct('x', pts{k}, 'fval', fval, 'info', info(1));
    end
    S.sigma_e_calibrated = M_.Sigma_e;
    M2 = set_all_parameters(pts{2}, estim_params_, M_);
    S.sigma_e_at_point2 = M2.Sigma_e; S.h_at_point2 = M2.H;
    R.(sprintf('stage%d', STAGE)) = S;
end
fid = fopen('ref_sc.json', 'w'); fprintf(fid, '%s', jsonencode(R)); fclose(fid);
