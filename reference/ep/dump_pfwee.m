% dump_pfwee.m -- perfect foresight with expectation errors reference for dynare-rust.
% Writes ref_pfwee_<case>.json: oo_.endo_simul, oo_.exo_simul, oo_.steady_state,
% oo_.exo_steady_state, oo_.pfwee.terminal_info / shocks_info and timings.
% Run from a PRIVATE copy of reference/ep:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_pfwee.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);
cases = {
    'learnt',   '-DCASE=1'
    'constant', '-DCASE=2'
    'datafile', '-DCASE=3'
    'endval',   '-DCASE=4'
};
global M_ options_ oo_
for c = 1:size(cases, 1)
    name = cases{c, 1};
    t0 = tic;
    eval(sprintf('dynare pfwee %s nointeractive nograph noclearall console', cases{c, 2}));
    S = struct();
    S.t_dynare_total = toc(t0);
    S.endo_names = M_.endo_names;
    S.endo_simul = oo_.endo_simul;
    S.exo_simul = oo_.exo_simul;
    S.steady_state = oo_.steady_state;
    S.exo_steady_state = oo_.exo_steady_state;
    S.terminal_info = oo_.pfwee.terminal_info;
    S.shocks_info = oo_.pfwee.shocks_info;
    fid = fopen(sprintf('ref_pfwee_%s.json', name), 'w');
    fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true));
    fclose(fid);
    fprintf('%s: %.3f s\n', name, S.t_dynare_total);
end
