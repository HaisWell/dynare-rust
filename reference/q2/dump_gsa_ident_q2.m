% dump_gsa_ident_q2.m -- sensitivity(identification=1) (Morris screening of the
% moments, map_identification.m) and sensitivity(identification=1, morris=2) on nk_q2.mod
% (CASE 5, 6): ref_q2_gsa_ident.json (SAvdec, SAcc, SAnorm), logs q2_nk_c<k>_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_gsa_ident_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_q2 -DCASE=5 nointeractive noclearall
copyfile('nk_q2.log', 'q2_nk_c5_matlab.log');
S = load(fullfile('nk_q2', 'gsa', 'screen', 'nk_q2_morris_IDE.mat'));
R = struct('SAvdec', S.SAvdec, 'SAcc', S.SAcc, 'SAnorm', S.SAnorm);
fid = fopen('ref_q2_gsa_ident.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_gsa_ident.json');
clear oo_ M_ options_ estim_params_ bayestopt_
dynare nk_q2 -DCASE=6 nointeractive noclearall
copyfile('nk_q2.log', 'q2_nk_c6_matlab.log');
