% dump_glue_q2.m -- sensitivity(rmse=1, glue=1) on nk_q2.mod (CASE 7): the GLUE file
% nk_q2/gsa/nk_q2_glue_prior.mat to ref_q2_glue.json, the log to q2_nk_c7_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_glue_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_q2 -DCASE=7 -DMATLAB_GLUE nointeractive noclearall
copyfile('nk_q2.log', 'q2_nk_c7_matlab.log');
G = load(fullfile('nk_q2', 'gsa', 'nk_q2_glue_prior.mat'));
R = struct();
R.Sam_data = G.Sam.data; R.Sam_dim = G.Sam.dim;
R.Obs_data = G.Obs.data;
R.Lik_names = {G.Lik.name}; R.Lik_data = {G.Lik.data};
R.Out_names = {G.Out.name}; R.Out_data = {G.Out.data};
R.order_var = G.Info.order_var;
fid = fopen('ref_q2_glue.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_glue.json');
