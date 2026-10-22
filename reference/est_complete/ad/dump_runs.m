% dump_runs.m -- Dynare 7.2 reference estimations of est_ad.mod (estimation completion):
% analytic_derivation with mode_compute 4/5/101, newrat's Hessian options 0/2,
% use_penalized_objective_for_hessian and cova_compute=0; the mode file (xparam1, hh, fval)
% and the Laplace approximation of each run, in ref_runs.json.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_runs.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
runs = {
    'ad_mc4',      '-DSTAGE=1 -DMC=4 -DAD=1'
    'ad_mc5',      '-DSTAGE=1 -DMC=5 -DAD=1'
    'ad_mc101',    '-DSTAGE=1 -DMC=101 -DAD=1'
    'ad_uni_mc4',  '-DSTAGE=2 -DMC=4 -DAD=1'
    'ad_exp_mc5',  '-DSTAGE=3 -DMC=5 -DAD=1'
    'ad_ml_mc4',   '-DSTAGE=4 -DMC=4 -DAD=1'
    'mc5_hess0',   '-DSTAGE=1 -DMC=5 -DHESS=0'
    'mc5_hess2',   '-DSTAGE=1 -DMC=5 -DHESS=2'
    'mc4',         '-DSTAGE=1 -DMC=4'
    'mc4_pen',     '-DSTAGE=1 -DMC=4 -DPEN=1'
    'mc4_cova0',   '-DSTAGE=1 -DMC=4 -DCOVA=0'
};
R = struct();
for k = 1:size(runs,1)
    cd(here);
    clear global
    try
        eval(sprintf('dynare est_ad %s nointeractive', runs{k,2}));
        global oo_ options_
        m = load(fullfile('est_ad','Output','est_ad_mode.mat'));
        S = struct('xparam1', m.xparam1, 'fval', m.fval, 'error', '');
        if isfield(m,'hh'), S.hh = m.hh; else, S.hh = []; end
        if isfield(oo_,'MarginalDensity') && isfield(oo_.MarginalDensity,'LaplaceApproximation')
            S.laplace = oo_.MarginalDensity.LaplaceApproximation;
        else
            S.laplace = NaN;
        end
        if isfield(oo_.posterior.optimization,'optimization_info')
            S.funcCount = oo_.posterior.optimization.optimization_info.funcCount;
            S.iterations = oo_.posterior.optimization.optimization_info.iterations;
        end
    catch e
        S = struct('error', e.message);
    end
    R.(runs{k,1}) = S;
    delete(fullfile('est_ad','Output','est_ad_mode.mat'));
end
cd(here);
fid = fopen('ref_runs.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
