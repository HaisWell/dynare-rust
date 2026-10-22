% dump_q1diag.m -- model_diagnostics output of q1_diag.mod (CASE 1..4), written
% to q1diag_<case>.log. Run from a PRIVATE copy of reference/q1:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q1diag.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
for c = 1:4
    try
        txt = evalc(sprintf('dynare q1_diag -DCASE=%d nointeractive nograph noclearall', c));
    catch err
        txt = getReport(err);
    end
    fid = fopen(sprintf('q1diag_%d.log', c), 'w');
    fprintf(fid, '%s', txt);
    fclose(fid);
end
