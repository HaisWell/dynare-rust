% dump_ident_files_q2.m -- identification(gsa_sample_file=1) after sensitivity and
% identification(load_ident_files=1) on nk_q2.mod (CASE 3, 4); logs q2_nk_c<k>_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_ident_files_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
for cs = [3 4]
    eval(sprintf('dynare nk_q2 -DCASE=%d nointeractive noclearall', cs));
    copyfile('nk_q2.log', sprintf('q2_nk_c%d_matlab.log', cs));
end
