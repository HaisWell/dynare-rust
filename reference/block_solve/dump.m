% dump.m -- model(block) reference for the Rust block-wise solvers (perfect foresight and steady).
% Writes ref_block_solve.json: for each case oo_.endo_simul, oo_.steady_state, the solver status,
% and the per-block status of solve_block_decomposed_problem.
%
% Run from a PRIVATE copy of reference/block_solve (Dynare writes +fname/ next to the .mod;
% sw07_pf_b.mod needs nothing else):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log.txt 2>&1
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);

% Dynare 7.1 itself fails (missing static block file / index error) on blk_fwd_pf_0 and
% blk_rbc_1_9; those are not run here.
cases = {
    'rbc_perm',        'rbc_b',          '-DCASE=1'
    'rbc_temp',        'rbc_b',          '-DCASE=2'
    'rbc_ssm',         'rbc_b',          '-DCASE=3'
    'rbc_predet',      'rbc_predet_b',   ''
    'q1_pf_lbj',       'q1_pf_b',        '-DCASE=1'
    'q1_pf_gmres',     'q1_pf_b',        '-DCASE=2'
    'blk_back_pf_1',   'blk_back_pf_1',  ''
    'blk_back_pf_8',   'blk_back_pf_8',  ''
    'blk_nk_pf_3',     'blk_nk_pf_3',    ''
    'blk_nk_pf_4',     'blk_nk_pf_4',    ''   % stack_solve_algo=4 (lnsrch1); BS_SET=a4
    'sw07',            'sw07_pf_b',      ''
    'blk_nk_ss_0',    'blk_nk_ss_0',    ''
    'blk_nk_ss_6',     'blk_nk_ss_6',    ''
    'blk_back_ss_0',   'blk_back_ss_0',  ''
    'blk_fwd_ss_10',   'blk_fwd_ss_10',  ''
};
% Dynare 7.1's block steady state fails when it follows a perfect-foresight case in the same
% MATLAB session (stale block packages), so the two sets are dumped by separate sessions:
%   BS_SET=pf (default) -> ref_block_solve.json, BS_SET=ss -> ref_block_solve_ss.json
set = getenv('BS_SET');
if isempty(set), set = 'pf'; end
isss = cellfun(@(s) ~isempty(strfind(s, '_ss_')), cases(:, 1));
isa4 = strcmp(cases(:, 1), 'blk_nk_pf_4');
if strcmp(set, 'ss'), cases = cases(isss, :); elseif strcmp(set, 'a4'), cases = cases(isa4, :); else, cases = cases(~isss & ~isa4, :); end
global M_ options_ oo_
R = struct();
for c = 1:size(cases, 1)
    name = cases{c, 1};
    clear functions; rehash;   % stale +fname/+block packages of earlier cases
    try
        if isempty(cases{c, 3})
            eval(sprintf('dynare %s nointeractive nograph noclearall', cases{c, 2}));
        else
            eval(sprintf('dynare %s %s nointeractive nograph noclearall', cases{c, 2}, cases{c, 3}));
        end
    catch ME
        % e.g. Dynare 7.1 looks for a static block file the preprocessor did not write
        R.(name) = struct('matlab_error', ME.message);
        fprintf('%s FAILED: %s\n', name, ME.message);
        continue
    end
    S = struct();
    S.endo_names = M_.endo_names;
    S.maximum_lag = M_.maximum_lag;
    S.maximum_lead = M_.maximum_lead;
    S.steady_state = oo_.steady_state;
    if isfield(oo_, 'endo_simul')
        S.endo_simul = oo_.endo_simul;
    end
    if isfield(oo_, 'deterministic_simulation') && isfield(oo_.deterministic_simulation, 'status')
        S.status = oo_.deterministic_simulation.status;
        S.error = oo_.deterministic_simulation.error;
        if isfield(oo_.deterministic_simulation, 'block')
            S.block_error = [oo_.deterministic_simulation.block.error];
        end
    end
    S.nblocks_dyn = length(M_.block_structure.block);
    R.(name) = S;
    fprintf('%s done\n', name);
end
fid = fopen(['ref_block_solve' strrep(['_' set], '_pf', '') '.json'], 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
