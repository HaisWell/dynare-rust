% gen_data.m -- simulate model_qpm_obs.csv from model_qpm.mod at its calibration.
%
% No real-world data: 95 quarters (2002Q1-2025Q3) of the eight observables from the first-order
% solution with Gaussian shocks of covariance Sigma_e (rng(2006, 'twister')), after a 200-quarter
% burn-in from the steady state (the unit roots start at 0). The survey series pie4_exp is
% missing in the first 8 quarters, as survey data often start late.
%
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('gen_data.m')"
fname = 'model_qpm'; seed = 2006; T = 95; missing = {'pie4_exp', 8};
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
txt = fileread([fname '.mod']);
cut = strfind(txt, '// ESTIMATION CALL');
txt = [txt(1:cut(1)-1) sprintf('\nstoch_simul(order=1, irf=0, periods=0, noprint, nograph, nomoments, nocorr);\n')];
fid = fopen([fname '_sim.mod'], 'w'); fprintf(fid, '%s', txt); fclose(fid);
eval(['dynare ' fname '_sim nointeractive']);
global M_ options_ oo_

burn = 200;
rng(seed, 'twister');
ex = randn(burn + T, M_.exo_nbr) * chol(M_.Sigma_e);
y = simult_(M_, options_, oo_.dr.ys, oo_.dr, ex, 1);
y = y(:, burn + 2:end);
names = options_.varobs;
D = zeros(T, numel(names));
for i = 1:numel(names)
    D(:, i) = y(strcmp(M_.endo_names, names{i}), :)';
end
for k = 1:size(missing, 1)
    D(1:missing{k, 2}, strcmp(names, missing{k, 1})) = NaN;
end
d0 = datetime(2002, 3, 31);
fid = fopen([fname '_obs.csv'], 'w');
fprintf(fid, ',%s\n', strjoin(names(:)', ','));
for t = 1:T
    dt = dateshift(d0 + calmonths(3*(t-1)), 'end', 'month');
    fprintf(fid, '%s', datestr(dt, 'yyyy-mm-dd'));
    for i = 1:numel(names)
        if isnan(D(t, i)), fprintf(fid, ','); else, fprintf(fid, ',%.10g', D(t, i)); end
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('%s_obs.csv written: %d quarters x %d series\n', fname, T, numel(names));
