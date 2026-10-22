% dump_datafiles.m -- initval_file / histval_file reference for dynare-rust.
% Writes ivf.xlsx (from ivf.csv) and ref_datafiles_<case>.json (oo_.endo_simul,
% oo_.exo_simul, M_.endo_histval). Run from a PRIVATE copy of reference/ep:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_datafiles.m')"
% The Dynare version is taken from the DYNARE_VERSION environment variable (default 7.2):
%   DYNARE_VERSION=7.1 /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
ver = getenv('DYNARE_VERSION');
if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
writetable(readtable('ivf.csv'), 'ivf.xlsx');
cases = {'initval_csv', 'initval_xlsx', 'histval', 'histval_ep', 'histval_long', 'histval_first_obs', 'histval_last_obs'};
global M_ options_ oo_
for c = 1:numel(cases)
    name = cases{c};
    eval(sprintf('dynare datafiles -DCASE=%d nointeractive nograph noclearall console', c));
    S = struct();
    S.endo_names = M_.endo_names;
    S.endo_simul = oo_.endo_simul;
    S.exo_simul = oo_.exo_simul;
    S.periods = options_.periods;
    S.endo_histval = M_.endo_histval;
    S.steady_state = oo_.steady_state;
    fid = fopen(sprintf('ref_datafiles_%s.json', name), 'w');
    fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true));
    fclose(fid);
    fprintf('%s done\n', name);
end

% error cases (Dynare 7.2 histval_file): CASE 8..13 -> ref_datafiles_errors.json
E = struct('case', {}, 'message', {});
for c = 8:13
    try
        eval(sprintf('dynare datafiles -DCASE=%d nointeractive nograph noclearall console', c));
        msg = '';
    catch err
        msg = err.message;
    end
    E(end+1) = struct('case', c, 'message', msg); %#ok<SAGROW>
    fprintf('error case %d: %s\n', c, msg);
end
fid = fopen('ref_datafiles_errors.json', 'w');
fprintf(fid, '%s', jsonencode(E));
fclose(fid);
