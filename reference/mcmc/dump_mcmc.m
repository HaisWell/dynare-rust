% Interoperability of the MCMC files between Dynare (DYNARE_VERSION, default 7.2) and dynare-rust .
% Run from a scratch folder (never inside a source checkout) that holds:
%   m1/  dynare-rust's stage-1 run of nk_mcmc.mod (folder nk_mcmc/ with metropolis/ and
%        Output/nk_mcmc_mode.mat) + nk_mcmc_matlab.mod copied as nk_mcmc.mod, nk_model.inc,
%        nk_data.csv
%   m2/  nk_mcmc_matlab.mod as nk_mcmc.mod, nk_model.inc, nk_data.csv (fresh Dynare run)
%   m3/  (second call, PHASE=2) dynare-rust's continuation of m2's chains
% PHASE 1:  m1: Dynare loads dynare-rust's chains (load_mh_file, mh_replic=0), then continues
%           them by 1000 draws; m2: Dynare's own fresh run (mode + 2 x 3000 draws), then a
%           reload. PHASE 2: m3: Dynare loads dynare-rust's continuation of its chains.
% Writes ref_mcmc_phase<PHASE>.json next to this script's working folder.
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "PHASE=1; run('dump_mcmc.m')"
if ~exist('PHASE', 'var')
    PHASE = 1;
end
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
root = pwd;
out = struct();
if PHASE == 1
    cd(fullfile(root, 'm1'));
    dynare nk_mcmc -DSTAGE=3 nolog noclearall nointeractive
    out.m1_load = mcmc_stats(M_, oo_, options_, estim_params_);
    dynare nk_mcmc -DSTAGE=2 -DREPLIC=1000 nolog noclearall nointeractive
    out.m1_cont = mcmc_stats(M_, oo_, options_, estim_params_);
    cd(fullfile(root, 'm2'));
    dynare nk_mcmc -DSTAGE=1 nolog noclearall nointeractive
    out.m2_fresh = mcmc_stats(M_, oo_, options_, estim_params_);
    dynare nk_mcmc -DSTAGE=3 nolog noclearall nointeractive
    out.m2_load = mcmc_stats(M_, oo_, options_, estim_params_);
else
    cd(fullfile(root, 'm3'));
    dynare nk_mcmc -DSTAGE=3 nolog noclearall nointeractive
    out.m3_load = mcmc_stats(M_, oo_, options_, estim_params_);
end
cd(root);
fid = fopen(sprintf('ref_mcmc_phase%d.json', PHASE), 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);

function s = mcmc_stats(M_, oo_, options_, estim_params_)
s.mean = get_posterior_parameters('mean', M_, estim_params_, oo_, options_);
s.median = get_posterior_parameters('median', M_, estim_params_, oo_, options_);
s.hpdinf = get_posterior_parameters('hpdinf', M_, estim_params_, oo_, options_);
s.hpdsup = get_posterior_parameters('hpdsup', M_, estim_params_, oo_, options_);
s.std = get_posterior_parameters('std', M_, estim_params_, oo_, options_);
s.mhm = oo_.MarginalDensity.ModifiedHarmonicMean;
record = load_last_mh_history_file([M_.dname filesep 'metropolis'], M_.fname);
s.MhDraws = record.MhDraws;
s.LastParameters = record.LastParameters;
s.LastLogPost = record.LastLogPost;
s.KeepedDraws = record.KeepedDraws;
s.AcceptanceRatio = record.AcceptanceRatio;
end
