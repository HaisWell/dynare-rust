% Simulated data for the analytic_derivation tests (y, c in logs; 200 periods), with small
% measurement errors. Run from a private copy of this folder.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare gen_data nointeractive
global oo_ M_
iy = find(strcmp(M_.endo_names,'y')); ic = find(strcmp(M_.endo_names,'c'));
rng(3);
Y = oo_.endo_simul([iy ic], :)';
Y(:,2) = Y(:,2) + 0.002*randn(size(Y,1),1);
Y(:,1) = Y(:,1) + 0.001*randn(size(Y,1),1);
fid = fopen('ad_data.csv','w'); fprintf(fid,'y,c\n'); fprintf(fid,'%.17g,%.17g\n',Y'); fclose(fid);
