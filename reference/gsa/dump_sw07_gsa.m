% dump_sw07_gsa.m -- dynare_sensitivity on the SW07 model (STAGE 20: stability map,
% STAGE 21: RMSE mapping), reference for tests/gsa_sw07.rs.
% Also STAGE 20 with ilptau=2 (the option added to the sensitivity call of a copy of
% sw07.mod): SW07 has 29 structural parameters, under the 30 above which Dynare scrambles the
% LPTAU columns with randperm, so ilptau=2 forces that branch.
% Run from a private copy of reference/sw07 with this file next to sw07.mod:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_sw07_gsa.m')"
% Writes ref_gsa_sw07_stage20.json, ref_gsa_sw07_stage21.json, ref_gsa_sw07_stage20_ilptau2.json.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
% sw07.mod with ilptau=2 in the STAGE 20 sensitivity call
txt = fileread('sw07.mod');
txt2 = strrep(txt, 'sensitivity(nodisplay, graph_format=none);', 'sensitivity(nodisplay, graph_format=none, ilptau=2);');
assert(~strcmp(txt, txt2), 'STAGE 20 sensitivity call not found');
if ~exist('stages', 'var')
    stages = {'20', '21', '20_ilptau2'};
end
for k = 1:numel(stages)
    clearvars -except k stages txt txt2 here
    cd(here);
    st = stages{k};
    t0 = tic;
    fid = fopen('sw07.mod', 'w');
    if strcmp(st, '20_ilptau2')
        fprintf(fid, '%s', txt2);
        stn = 20;
    else
        fprintf(fid, '%s', txt);
        stn = str2double(st);
    end
    fclose(fid);
    if exist(fullfile('sw07', 'gsa'), 'dir'), rmdir(fullfile('sw07', 'gsa'), 's'); end
    try
        eval(sprintf('dynare sw07 -DSTAGE=%d noclearall nolog nointeractive', stn));
    catch err
        % monte_carlo_filtering.m can fail at disp(vvarvecm{invar}) when more than one series
        % has a constant RMSE; the sample, RMSEs and likelihoods are saved before that point.
        fprintf('Dynare error (continuing with the saved results): %s\n', err.message);
    end
    cd(here);
    fprintf('STAGE %s: %.1f s\n', st, toc(t0));
    d = load(fullfile('sw07', 'gsa', 'sw07_prior.mat'));
    out = struct();
    keep = {'lpmat', 'lpmat0', 'istable', 'iunstable', 'iindeterm', 'iwrong', 'ixun', 'infox', ...
            'nspred', 'nboth', 'nfwrd', 'rmse_MC', 'r2_MC', 'logpo2', 'likelihood'};
    for j = 1:numel(keep)
        if isfield(d, keep{j}), out.(keep{j}) = d.(keep{j}); end
    end
    if stn == 20
        Nsam = size(d.lpmat, 1);
        itot = 1:Nsam;
        itmp = itot(~ismember(itot, d.istable));
        out.ks_stable = ks_all(d.lpmat, d.istable, itmp);
    end
    fid = fopen(sprintf('ref_gsa_sw07_stage%s.json', st), 'w');
    fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
    fclose(fid);
end
fid = fopen('sw07.mod', 'w'); fprintf(fid, '%s', txt); fclose(fid);

function out = ks_all(lpmat, ib, inb)
% as gsa.stability_mapping_univariate(..., iplot=0)
proba = NaN(size(lpmat,2),1); dproba = proba;
for j=1:size(lpmat,2)
    [~, proba(j), dproba(j)] = gsa.smirnov_test(lpmat(ib,j), lpmat(inb,j));
end
out = struct('proba', proba, 'dproba', dproba);
end
