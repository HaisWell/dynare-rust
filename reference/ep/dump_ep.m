% dump_ep.m -- extended path reference for dynare-rust's `ep` module.
% Writes ref_ep.json: for each case the shocks Dynare drew (oo_.exo_simul), the simulated
% paths (oo_.endo_simul), the steady state, the status and timings (the extended_path call
% re-run on the same shocks, best of 3).
%
% Run from a PRIVATE copy of reference/ep (Dynare writes +rbcii/ next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_ep.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);

cases = {
    'ep0',          '-DCASE=1'
    'sep_tensor1',  '-DCASE=2'
    'sep_tensor2',  '-DCASE=3'
    'sep_sparse2',  '-DCASE=4'
    'sep_stroud1',  '-DCASE=5'
    'sep_unscented2', '-DCASE=6'
    'sep_hybrid1',  '-DCASE=7'
    'ep0_first_order_guess', '-DCASE=8'
};
if exist('only_cases', 'var')
    cases = cases(ismember(cases(:,1), only_cases), :);
end
global M_ options_ oo_

for c = 1:size(cases, 1)
    name = cases{c, 1};
    t0 = tic;
    eval(sprintf('dynare rbcii %s nointeractive nograph noclearall console', cases{c, 2}));
    t_total = toc(t0);
    S = struct();
    S.endo_names = M_.endo_names;
    S.exo_names = M_.exo_names;
    S.steady_state = oo_.steady_state;
    S.Sigma_e = M_.Sigma_e;
    S.endo_simul = oo_.endo_simul;
    S.exo_simul = oo_.exo_simul;
    S.status = oo_.extended_path.status;
    S.ep = options_.ep;
    S.t_dynare_total = t_total;
    % the extended_path call alone on the same shocks
    shocks = oo_.exo_simul(2:end, :);
    T = size(shocks, 1);
    best = inf;
    for r = 1:3
        t1 = tic;
        [ts, oo1] = extended_path([], T, shocks, options_, M_, oo_);
        best = min(best, toc(t1));
    end
    S.t_ep = best;
    S.max_diff_rerun = max(abs(oo1.endo_simul(:) - oo_.endo_simul(:)));
    S.binding_periods = sum(oo_.endo_simul(strcmp(M_.endo_names, 'mu'), :) > 1e-8);
    fprintf('%s: total %.3f s, ep %.4f s, status %d, binding %d, rerun diff %g\n', name, t_total, best, S.status, S.binding_periods, S.max_diff_rerun);
    % one file per case (a MATLAB crash in a later case keeps the earlier ones)
    fid = fopen(sprintf('ref_ep_%s.json', name), 'w');
    fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true));
    fclose(fid);
end
