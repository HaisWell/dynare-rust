ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
rng(3);
dynare nk_gen nointeractive
s = oo_.endo_simul(1:3, 2:end)';
fid = fopen('nk_data.csv', 'w');
fprintf(fid, 'y,pi,i\n');
fprintf(fid, '%.17g,%.17g,%.17g\n', s');
fclose(fid);
