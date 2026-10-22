% gen_data.m -- simulate the observation data sw07_obs.csv from sw07.mod at its calibration.
%
% No real-world data: 95 quarters (2002Q1-2025Q3) of the nine observables, drawn from the
% first-order solution with Gaussian shocks of covariance Sigma_e (MATLAB rng(2007, 'twister')),
% after a 200-quarter burn-in that starts at the steady state (the two random walks, pistar and
% labtrend, start at 0 at the beginning of the burn-in). Survey series start late, as in real
% data: pinfexp4obs and pinflrobs are missing in the first 4 quarters, dw in the first one.
%
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('gen_data.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
txt = fileread('sw07.mod');
cut = strfind(txt, '// ESTIMATION CALL');
txt = [txt(1:cut(1)-1) sprintf('\nstoch_simul(order=1, irf=0, periods=0, noprint, nograph, nomoments, nocorr);\n')];
fid = fopen('sw07_sim.mod', 'w'); fprintf(fid, '%s', txt); fclose(fid);
dynare sw07_sim nointeractive
global M_ options_ oo_

burn = 200; T = 95;
rng(2007, 'twister');
ex = randn(burn + T, M_.exo_nbr) * chol(M_.Sigma_e + 1e-300*eye(M_.exo_nbr));
y = simult_(M_, options_, oo_.dr.ys, oo_.dr, ex, 1);   % endo_nbr x (burn+T+1), column 1 = y0
y = y(:, burn + 2:end);

names = options_.varobs;
D = zeros(T, numel(names));
for i = 1:numel(names)
    D(:, i) = y(strcmp(M_.endo_names, names{i}), :)';
end
D(1:4, strcmp(names, 'pinfexp4obs')) = NaN;
D(1:4, strcmp(names, 'pinflrobs')) = NaN;
D(1, strcmp(names, 'dw')) = NaN;

% quarter-end dates, as in the other reference data files
d0 = datetime(2002, 3, 31);
fid = fopen('sw07_obs.csv', 'w');
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
fprintf('sw07_obs.csv written: %d quarters x %d series\n', T, numel(names));
