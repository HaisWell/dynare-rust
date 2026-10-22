% dump_deterministic.m -- perfect-foresight reference for the Rust `deterministic` module.
% Writes ref_deterministic.json: for each case oo_.endo_simul (n x (maxlag+T+maxlead)),
% oo_.exo_simul, oo_.steady_state, the solver status and timings.
%
% Run from a PRIVATE copy of reference/deterministic (Dynare writes +fname/ next to the
% .mod), no data file needed:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_deterministic.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);

cases = {
    'rbc_perm',     'rbc',    '-DCASE=1'
    'rbc_temp',     'rbc',    '-DCASE=2'
    'rbc_ssm',      'rbc',    '-DCASE=3'
    'rbc_linapprox','rbc',    '-DCASE=4'
    'rbc_histval',  'rbc',    '-DCASE=5'
    'rbc_homotopy', 'rbc',    '-DCASE=6'
    'rbc_predet',   'rbc_predet', ''
    'mshocks',      'mshocks', ''
    'linear',       'linear', ''
    'sw07',         'sw07_pf', ''
};
global M_ options_ oo_
R = struct();
for c = 1:size(cases, 1)
    name = cases{c, 1};
    t0 = tic;
    if isempty(cases{c, 3})
        eval(sprintf('dynare %s nointeractive nograph noclearall', cases{c, 2}));
    else
        eval(sprintf('dynare %s %s nointeractive nograph noclearall', cases{c, 2}, cases{c, 3}));
    end
    t_total = toc(t0);

    S = struct();
    S.endo_names = M_.endo_names;
    S.exo_names = M_.exo_names;
    S.maximum_lag = M_.maximum_lag;
    S.maximum_lead = M_.maximum_lead;
    S.endo_simul = oo_.endo_simul;
    S.exo_simul = oo_.exo_simul;
    S.steady_state = oo_.steady_state;
    S.exo_steady_state = oo_.exo_steady_state;
    S.status = oo_.deterministic_simulation.status;
    S.error = oo_.deterministic_simulation.error;
    if isfield(oo_.deterministic_simulation, 'iterations')
        S.iterations = oo_.deterministic_simulation.iterations;
    end
    S.t_dynare_total = t_total;
    % timing of the solver alone (setup + solver, no printing), best of 5
    opts = options_; opts.verbosity = 0;
    % noprint breaks homotopy_loop in Dynare 7.1 (iter_summary_table undefined)
    opts.noprint = isempty(strfind(name, 'homotopy'));
    oo0 = oo_;
    if isfield(oo0, 'initial_steady_state') && ~isempty(oo0.initial_steady_state)
        % keep the terminal steady state of the run
    end
    best = inf;
    for r = 1:5
        t1 = tic;
        oo1 = perfect_foresight_setup(M_, opts, oo0);
        oo1 = perfect_foresight_solver(M_, opts, oo1);
        best = min(best, toc(t1));
    end
    S.t_solver = best;
    R.(name) = S;
    fprintf('%s: total %.3f s, solver %.4f s, status %d\n', name, t_total, best, S.status);
end
fid = fopen('ref_deterministic.json', 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
