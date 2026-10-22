% Run from a private copy of reference/gaps:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log.txt 2>&1
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
dynare gap_me noclearall nolog
r = struct();
r.HPDinf_ME = oo_.forecast.HPDinf_ME; r.HPDsup_ME = oo_.forecast.HPDsup_ME;
r.HPDinf = oo_.forecast.HPDinf; r.HPDsup = oo_.forecast.HPDsup; r.Mean = oo_.forecast.Mean;
fid = fopen('ref_gap_me.json', 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
