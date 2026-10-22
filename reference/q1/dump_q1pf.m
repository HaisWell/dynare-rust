% dump_q1pf.m -- perfect-foresight / steady-state reference (stack_solve_algo
% variants, lmmcp, purely backward/forward/static models). Writes ref_q1pf.json: for each
% case oo_.endo_simul, oo_.steady_state, the solver status and error.
%
% Run from a PRIVATE copy of reference/q1:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q1pf.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
cases = {
    'pf_ssa1',   'q1_pf',   '-DCASE=1'
    'pf_ssa2',   'q1_pf',   '-DCASE=2'
    'pf_ssa3ilu','q1_pf',   '-DCASE=3'
    'pf_ssa2bd', 'q1_pf',   '-DCASE=4'
    'pf_ssa6',   'q1_pf',   '-DCASE=5'
    'pf_ssa7',   'q1_pf',   '-DCASE=6'
    'pf_robust', 'q1_pf',   '-DCASE=7'
    'pf_ssa3',   'q1_pf',   '-DCASE=8'
    'pf_ssa7d',  'q1_pf',   '-DCASE=9'
    'pf_mlf',    'q1_pf',   '-DCASE=10'
    'sp',        'q1_sp',   ''
    'mcp_pf',    'q1_mcp',  '-DCASE=1'
    'mcp_ss',    'q1_mcp',  '-DCASE=2'
    'mcp_pf7',   'q1_mcp',  '-DCASE=3'
    'back',      'q1_back', '-DCASE=1'
    'forward',   'q1_back', '-DCASE=2'
    'static',    'q1_back', '-DCASE=3'
};
global M_ options_ oo_
R = struct();
for c = 1:size(cases, 1)
    name = cases{c, 1};
    S = struct();
    try
        eval(sprintf('dynare %s %s nointeractive nograph noclearall', cases{c, 2}, cases{c, 3}));
        S.endo_names = M_.endo_names;
        S.steady_state = oo_.steady_state;
        if isfield(oo_, 'endo_simul') && ~isempty(oo_.endo_simul) && isfield(oo_, 'deterministic_simulation') ...
                && isfield(oo_.deterministic_simulation, 'status')
            S.endo_simul = oo_.endo_simul;
            S.exo_simul = oo_.exo_simul;
            S.exo_steady_state = oo_.exo_steady_state;
            S.status = oo_.deterministic_simulation.status;
            S.error = oo_.deterministic_simulation.error;
        end
    catch err
        S.failure = getReport(err);
        disp(S.failure);
    end
    R.(name) = S;
    clear -global oo_
    global M_ options_ oo_
end
fid = fopen('ref_q1pf.json', 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
