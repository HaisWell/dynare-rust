% dump.m -- extended path with options_.ep.solve_algo = 10 (lmmcp). Run from a PRIVATE copy:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);
cases = {'sep1','-DCASE=1'; 'sep2sparse_hybrid','-DCASE=2'; 'ep0','-DCASE=3'};
global M_ options_ oo_
for c = 1:size(cases, 1)
    name = cases{c, 1};
    eval(sprintf('dynare ep_lmmcp %s nointeractive nograph noclearall console', cases{c, 2}));
    % MATLAB's own shocks, then the same call with lmmcp on the tree
    shocks = oo_.exo_simul(2:end, :);
    T = size(shocks, 1);
    options_.ep.solve_algo = 10;
    [ts, oo1] = extended_path([], T, shocks, options_, M_, oo_);
    S = struct();
    S.endo_names = M_.endo_names;
    S.steady_state = oo_.steady_state;
    S.exo_simul = [oo_.exo_steady_state'; shocks];
    S.endo_simul = oo1.endo_simul;
    S.status = oo1.extended_path.status;
    S.binding_periods = sum(oo1.endo_simul(strcmp(M_.endo_names, 'mu'), :) > 1e-8);
    S.binding_r = sum(oo1.endo_simul(strcmp(M_.endo_names, 'r'), :) < 1e-8);
    fprintf('%s: status %d, binding mu %d, r %d\n', name, S.status, S.binding_periods, S.binding_r);
    fid = fopen(sprintf('ref_%s.json', name), 'w');
    fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true));
    fclose(fid);
end
