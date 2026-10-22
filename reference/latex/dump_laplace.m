% dump_laplace.m -- the pieces of Dynare's Laplace approximation on latex_nk.mod and
% latex_nk_alt.mod (mode_compute=4): the csminwel mode xparam1, fval, the Hessian hh of
% hessian.m at that mode (as saved in <fname>_mode.mat), the Laplace approximation, and
% hessian.m / dsge_likelihood re-evaluated at that mode. Lets tests/latex.rs separate the
% optimizer endpoint from the Hessian computation.
% Run from a PRIVATE copy of reference/latex:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_laplace.m')"
% Writes ref_laplace.json.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
R = struct();
for m = {'latex_nk_alt', 'latex_nk'}
    name = m{1};
    eval(sprintf('dynare %s nointeractive', name));
    cd(here);
    global M_ options_ oo_ estim_params_ bayestopt_
    mf = load(fullfile(name, 'Output', [name '_mode.mat']));
    S = struct('xparam1', mf.xparam1, 'hh', mf.hh, 'fval', mf.fval, 'laplace', oo_.MarginalDensity.LaplaceApproximation, ...
               'parameter_names', {mf.parameter_names}, 'gstep', options_.gstep);
    R.(name) = S;
end
fid = fopen('ref_laplace.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
