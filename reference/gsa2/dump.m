% dump.m -- GSA completion references (tests/gsa2.rs). Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log.txt 2>&1
% (under -batch the diary stays empty: ref_gsa2_stage10.log is the STAGE 10 part of log.txt)
% STAGE 10: estimation (short MCMC) then sensitivity(ppost=1, rmse=1)
% STAGE 11: sensitivity(rmse=1) and the same with load_rmse=1 (reloads the saved results)
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
% 11 first: STAGE 10 deletes nothing of it
if ~exist('stages', 'var'), stages = [11 10]; end
for st = stages
    clearvars -except st stages
    close all
    logf = sprintf('ref_gsa2_stage%d.log', st);
    if exist(logf, 'file'), delete(logf); end
    diary(logf);
    eval(sprintf('dynare nk_gsa -DSTAGE=%d -DDYNARE_MATLAB noclearall nolog', st));
    diary off
    out = struct();
    if st == 10
        d = load(fullfile('nk_gsa', 'gsa', 'nk_gsa_post.mat'));
        for f = fieldnames(d)'
            if ~strcmp(f{1}, 'vvarvecm'), out.(f{1}) = d.(f{1}); end
        end
        pf = dir(fullfile('nk_gsa', 'metropolis', 'nk_gsa_param*.mat'));
        for k = 1:numel(pf)
            if isempty(strfind(pf(k).name, 'param_irf'))
                p = load(fullfile('nk_gsa', 'metropolis', pf(k).name)); out.param_stock = p.stock; out.param_logpo = p.stock_logpo;
            end
        end
        if exist(fullfile('nk_gsa','Output','nk_gsa_mean.mat'),'file')
            m = load(fullfile('nk_gsa','Output','nk_gsa_mean.mat')); out.xparam1_mean = m.xparam1;
        end
    else
        d = load(fullfile('nk_gsa', 'gsa', 'nk_gsa_prior.mat'));
        for f = {'x','logpo2','likelihood','rmse_MC','r2_MC'}
            if isfield(d, f{1}), out.(f{1}) = d.(f{1}); end
        end
    end
    fid = fopen(sprintf('ref_gsa2_stage%d.json', st), 'w');
    fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
    fclose(fid);
end
