% dump_occbin_post.m -- Dynare 7.2: posterior distributions of the OccBin-smoothed variables
% (occbin_post.mod -DMCMC) -> ref_q3_occbin_post.json (posterior means, statistical check).
% Run from a PRIVATE COPY of reference/q3:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_occbin_post.m')"
% DYNARE_VERSION (environment) selects another Dynare install (default 7.2).
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = pwd;
dynare occbin_post -DMCMC nolog nointeractive
R = struct('posterior_mean', oo_.posterior_mean, 'SmoothedVariables_Mean', oo_.SmoothedVariables.Mean, ...
           'SmoothedShocks_Mean', oo_.SmoothedShocks.Mean);
fid = fopen(fullfile(here, 'ref_q3_occbin_post.json'), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dump written');
