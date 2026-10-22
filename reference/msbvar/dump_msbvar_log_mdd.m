% Dumps the deterministic marginal-data-density pieces of sbvar on msbvar_log.mod (Dynare 7.2 by default, see DYNARE_VERSION)
% into ref_msbvar_log_mdd.json at full precision: log p(a0hat,a+hat) (vlog_a0p),
% log p(Y|ahat) (vlog_Y_a) and log p(a+hat|a0hat,Y) (vlog_ap_Ya0).
% Runs in a temporary directory with a shadowed ms_mardd.m that saves them and returns
% before the (random) Gibbs part.
here = fileparts(mfilename('fullpath'));
work = tempname; mkdir(work);
copyfile(fullfile(here, 'msbvar_log.mod'), work);
copyfile(fullfile(here, 'msbvar_data.csv'), work);
cd(work);
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
src = fileread(fullfile(droot, 'matlab/ms-sbvar/ms_mardd.m'));
k = strfind(src, 'vlog_ap_Ya0 = ');
e = k(1) + find(src(k(1):end) == newline, 1) - 1;
ins = ['save(getenv(''MSBVAR_WS''), ''vlog_a0p'', ''vlog_Y_a'', ''vlog_ap_Ya0'');' newline 'return' newline];
fid = fopen(fullfile(work, 'ms_mardd.m'), 'w'); fwrite(fid, [src(1:e) ins src(e+1:end)]); fclose(fid);
addpath(fullfile(droot, 'matlab'));
ws = fullfile(work, 'mdd_ws.mat'); setenv('MSBVAR_WS', ws);
dynare('msbvar_log', 'noclearall', 'nograph');
out = load(ws);
fid = fopen(fullfile(here, 'ref_msbvar_log_mdd.json'), 'w');
fprintf(fid, '{"vlog_a0p": %.17g, "vlog_Y_a": %.17g, "vlog_ap_Ya0": %.17g}\n', out.vlog_a0p, out.vlog_Y_a, out.vlog_ap_Ya0);
fclose(fid);
close all
cd(here);
