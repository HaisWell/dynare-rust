% Dumps sbvar results (Dynare 7.2 by default, see DYNARE_VERSION) with `restriction equation` lines (msbvar_lin, msbvar_lin2) into ref_msbvar_lin.json.
% Shadows ms_mardd.m with a copy that saves the deterministic quantities and returns before
% the (random) Gibbs-sampler part of the marginal data density.
here = fileparts(mfilename('fullpath'));
cd(here);
shadow = here;
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
src = fileread(fullfile(droot, 'matlab/ms-sbvar/ms_mardd.m'));
k = strfind(src, 'vlog_ap_Ya0 = ');
e = k(1) + find(src(k(1):end) == newline, 1) - 1;
ins = ['wsfile = getenv(''MSBVAR_WS''); save(wsfile, ''A0hat'',''Fhat'',''ghat'',''Bhat'',''A0hatinv'',''xhat'',''fhat'',''imfhat'',''yforehat'',''Pmat'',''H0inv'',''Hpinv'',''n0'',''np'',''fss'',''vlog_a0p'',''vlog_Y_a'',''logMarLHres'',''vlog_ap_Ya0'',''Uiconst'',''Viconst'',''nvar'');' newline 'return' newline];
ins0 = ['wsfile = getenv(''MSBVAR_WS''); save(wsfile, ''A0hat'',''Fhat'',''ghat'',''Bhat'',''A0hatinv'',''xhat'',''fhat'',''imfhat'',''yforehat'',''Pmat'',''H0inv'',''Hpinv'',''n0'',''np'',''fss'',''Uiconst'',''Viconst'',''nvar'');' newline];
k0 = strfind(src, 'vlog_a0p = 0;'); e0 = k0(1) - 1;
% the second save (with the marginal-density pieces) is only reached with the Bayesian prior
src2 = [src(1:e0) ins0 src(e0+1:e) ins src(e+1:end)];
fid = fopen(fullfile(shadow,'ms_mardd.m'),'w'); fwrite(fid, src2); fclose(fid);
addpath(fullfile(droot, 'matlab')); 
cases = {'msbvar_lin','msbvar_lin2'};
out = struct();
for c = 1:numel(cases)
    ws = fullfile(tempdir, [cases{c} '_ws.mat']); setenv('MSBVAR_WS', ws);
    try, dynare(cases{c}, 'noclearall', 'nograph'); catch ME, disp(ME.message); end % no_bayesian_prior: Dynare's mdd code errors after the ML part
    out.(cases{c}) = load(ws);
    close all
end
fid = fopen('ref_msbvar_lin.json','w'); fwrite(fid, jsonencode(out, 'PrettyPrint', true)); fclose(fid);
delete(fullfile(here,'ms_mardd.m'));
