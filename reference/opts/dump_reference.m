% dump_reference.m -- Dynare 7.2 reference values for estimation options:
%   model_rbc.mod STAGE 1 (loglinear), 2 (loglinear + logdata), 3 (loglinear +
%   endogenous_prior) -> ref_rbc_<stage>.json; ../ml/model_ml.mod with -DPREFILTER
%   (prefilter=1 + endogenous_prior) -> ref_ml_prefilter.json.
% Run from a PRIVATE COPY of reference/ (needs ../ml):
%   cd opts; /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_reference.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
global M_ options_ oo_ estim_params_ bayestopt_
runs = {{here, 'model_rbc', '-DSTAGE=1', 'ref_rbc_1.json'}, {here, 'model_rbc', '-DSTAGE=2', 'ref_rbc_2.json'}, ...
        {here, 'model_rbc', '-DSTAGE=3', 'ref_rbc_3.json'}, {fullfile(here, '..', 'ml'), 'model_ml', '-DPREFILTER', 'ref_ml_prefilter.json'}};
for r = 1:numel(runs)
    cd(runs{r}{1});
    eval(sprintf('dynare %s %s nointeractive', runs{r}{2}, runs{r}{3}));
    R = struct();
    R.SmoothedVariables = oo_.SmoothedVariables;
    [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    R.xparam_init = xparam_init; R.steady_state = oo_.steady_state; R.data = dataset_.data;
    M2 = set_all_parameters(xparam_init, estim_params_, M_);
    [T, Rm, SS, info, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
    R.ghx = dr.ghx; R.ghu = dr.ghu; R.T = T; R.R = Rm;
    pts = {xparam_init, xparam_init .* [1.02; 0.97; 1.1; 0.9; ones(numel(xparam_init) - 4, 1)], xparam_init * 0.95};
    R.points = {};
    for k = 1:numel(pts)
        [fval, info] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        R.points{end+1} = struct('x', pts{k}, 'minus_loglik', fval, 'info', info(1));
    end
    fid = fopen(fullfile(here, runs{r}{4}), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
end
disp('dumps written');
