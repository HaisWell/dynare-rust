% dump_gsa.m -- reference values of dynare_sensitivity on nk_gsa.mod (tests/gsa_nk.rs).
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_gsa.m')"
% Writes nk_gsa_data.csv (STAGE 0) and ref_gsa_stage<k>.json, ref_gsa_stage<k>.log.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
if ~exist('stages', 'var')
    stages = [0 1 2 3 4 5 6];
end
for st = stages
    clearvars -except st stages
    close all
    logf = sprintf('ref_gsa_stage%d.log', st);
    if exist(logf, 'file'), delete(logf); end
    diary(logf);
    eval(sprintf('dynare nk_gsa -DSTAGE=%d -DDYNARE_MATLAB noclearall nolog', st));
    diary off
    out = struct();
    if st == 0
        iv = [find(strcmp(M_.endo_names, 'y')), find(strcmp(M_.endo_names, 'pi')), find(strcmp(M_.endo_names, 'i'))];
        Y = oo_.endo_simul(iv, :)';
        fid = fopen('nk_gsa_data.csv', 'w');
        fprintf(fid, 'y,pi,i\n');
        fprintf(fid, '%.17g,%.17g,%.17g\n', Y');
        fclose(fid);
        continue
    end
    if st == 4
        d = load(fullfile('nk_gsa', 'gsa', 'screen', 'nk_gsa_prior.mat'));
    else
        d = load(fullfile('nk_gsa', 'gsa', 'nk_gsa_prior.mat'));
    end
    keep = {'lpmat', 'lpmat0', 'istable', 'iunstable', 'iindeterm', 'iwrong', 'ixun', 'infox', ...
            'irestriction', 'nspred', 'nboth', 'nfwrd', 'rmse_MC', 'r2_MC', 'logpo2', 'likelihood', ...
            'x', 'rmse_mode', 'r2_mode'};
    for k = 1:numel(keep)
        if isfield(d, keep{k}), out.(keep{k}) = d.(keep{k}); end
    end
    if isfield(d, 'egg'), out.egg_abs = abs(d.egg); end
    if isfield(d, 'T') && ~isempty(d.T)
        % entries used by the reduced-form stages, in DR order
        oo_.dr = set_state_space(struct(), M_); ov = oo_.dr.order_var;
        nspred = M_.nspred;
        out.T_rows = {};
        names = {'y', 'pi'};
        for a = 1:2
            ie = find(strcmp(M_.endo_names(ov), names{a}));
            out.(['T_' names{a}]) = squeeze(d.T(ie, :, :))';   % nstable x (nspred+nexo)
        end
        out.state_names = M_.endo_names(ov(M_.nstatic+1:M_.nstatic+nspred));
    end
    pnames = [M_.exo_names(estim_params_.var_exo(:,1)); M_.param_names(estim_params_.param_vals(:,1))];
    % KS / correlation statistics of the stability map, as stability_mapping computes them
    if isfield(d, 'istable') && st <= 2
        Nsam = size(d.lpmat, 1);
        itot = 1:Nsam;
        itmp = itot(~ismember(itot, d.istable));
        [proba, dproba] = ks_all(d.lpmat, d.istable, itmp);
        out.ks_stable = struct('proba', proba, 'dproba', dproba);
        isolve = itot(~ismember(itot, d.iwrong));
        if ~isempty(d.iindeterm)
            itmp = isolve(~ismember(isolve, d.iindeterm));
            [proba, dproba] = ks_all(d.lpmat, itmp, d.iindeterm);
            out.ks_indeterm = struct('proba', proba, 'dproba', dproba);
        end
        if ~isempty(d.ixun)
            itmp = isolve(~ismember(isolve, d.ixun));
            [proba, dproba] = ks_all(d.lpmat, itmp, d.ixun);
            out.ks_unstable = struct('proba', proba, 'dproba', dproba);
        end
        [c0, pv] = corrcoef(d.lpmat(d.istable, :));
        out.corr_stable = struct('c', c0, 'p', pv);
    end
    if st == 4
        % elementary effects as reduced_form_screening computes them
        oo_.dr = set_state_space(struct(), M_); ov = oo_.dr.order_var;
        kn = size(d.lpmat, 1);
        np = size(d.lpmat, 2);
        out.morris = struct();
        ie = find(strcmp(M_.endo_names(ov), 'y'));
        iexo = find(strcmp(M_.exo_names, 'e_i'));
        y0 = gsa.teff(d.T(ie, iexo + M_.nspred, :), kn, d.istable);
        [SAmeas, SAMorris] = gsa.Morris_Measure_Groups(np + size(d.lpmat0, 2), [d.lpmat0 d.lpmat], y0, 6);
        out.morris.y_vs_e_i = struct('y0', y0, 'SAMorris', SAMorris);
    end
    fid = fopen(sprintf('ref_gsa_stage%d.json', st), 'w');
    fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
    fclose(fid);
end

function [proba, dproba] = ks_all(lpmat, ib, inb)
% as gsa.stability_mapping_univariate(..., iplot=0)
proba = NaN(size(lpmat,2),1); dproba = proba;
for j=1:size(lpmat,2)
    [~, proba(j), dproba(j)] = gsa.smirnov_test(lpmat(ib,j), lpmat(inb,j));
end
end
