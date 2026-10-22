% dump_occbin_mode.m -- mode finding (mode_compute=4, csminwel) of nk_est.mod with the OccBin
% piecewise Kalman filter (-DMODE) and with the inversion filter (-DMODE -DIVF); writes the
% modes, fvals and Hessians to ref_occbin_mode.json. Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_mode.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
variants = { '-DMODE', 'pkf'; '-DMODE -DIVF', 'ivf' };
out = struct();
for v = 1:size(variants, 1)
    clear functions
    tic;
    eval(sprintf('dynare nk_est %s noclearall nograph nolog', variants{v, 1}));
    secs = toc;
    global M_ oo_ bayestopt_
    m = load(fullfile('nk_est', 'Output', 'nk_est_mode.mat'));
    out.(variants{v, 2}) = struct('xparam1', m.xparam1, 'fval', m.fval, 'hh', m.hh, ...
        'names', {bayestopt_.name}, 'laplace', oo_.MarginalDensity.LaplaceApproximation, 'seconds', secs);
end
fid = fopen(fullfile(here, 'ref_occbin_mode.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin_mode.json');
