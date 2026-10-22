% dump_prior.m -- Dynare's `prior moments(distribution), irfs(distribution)`
% and `prior optimize` on rbc_q2.mod (CASE 9); summary statistics of the implied prior
% draws to ref_q2_prior.json, the log to q2_c9_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_prior.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare rbc_q2 -DCASE=9 nointeractive
copyfile('rbc_q2.log', 'q2_c9_matlab.log');
S = load(fullfile('rbc_q2', 'prior', 'rbc_q2_endogenous_variables_prior_draws.mat'));
F = S.FirstOrderMoments; V = S.SecondOrderMoments;
F(:, any(isnan(F), 1)) = [];
V(:, :, squeeze(any(any(isnan(V), 1), 2))) = [];
R = struct();
R.n_draws = size(F, 2);
R.mean_of_mean = mean(F, 2); R.var_of_mean = mean((F - mean(F, 2)).^2, 2);
R.mean_of_var = mean(V, 3); R.var_of_var = mean((V - mean(V, 3)).^2, 3);
I = S.ImpulseResponseFunctions.ea;
I(:, :, squeeze(any(any(isnan(I), 1), 2))) = [];
R.irf_ea_mean = mean(I, 3); R.irf_ea_std = std(I, 0, 3);
fid = fopen('ref_q2_prior.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_prior.json');
