% gen_data.m -- simulates nk_est.mod's piecewise-linear model (OccBin solver, one surprise
% shock vector per period, MATLAB's rng(1)) and writes nk_est_data.csv (levels of
% y, pi, i, inot). Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('gen_data.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_est -DGEN noclearall nograph nolog
global M_ options_ oo_
T = 120;
rng(1);
sd = sqrt(diag(M_.Sigma_e))';
shocks = randn(T, M_.exo_nbr) .* sd;
options_.occbin.simul.SHOCKS = shocks;
options_.occbin.simul.exo_pos = 1:M_.exo_nbr;
options_.occbin.simul.periods = T;
options_.occbin.simul.check_ahead_periods = 200;
[~, out] = occbin.solver(M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
assert(out.error_flag == 0);
names = {'y', 'pi', 'i', 'inot'};
idx = cellfun(@(n) find(strcmp(M_.endo_names, n)), names);
Z = out.piecewise(1:T, idx);
fid = fopen('nk_est_data.csv', 'w');
fprintf(fid, '%s\n', strjoin(names, ','));
for t = 1:T
    fprintf(fid, '%.17g,%.17g,%.17g,%.17g\n', Z(t, :));
end
fclose(fid);
fprintf('ZLB binds in %d of %d periods\n', sum(abs(Z(:, 3) - M_.params(strcmp(M_.param_names, 'ilb'))) < 1e-14), T);
