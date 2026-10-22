% dump_solveopt.m -- reference values for tests/estimation_solveopt.rs: Dynare's solvopt.m
% (mode_compute=101: numerical gradient, no constraints, options_.solveopt defaults with
% verbosity 0) on rosen5 (n = 2, 4) and quad5, written to ref_solveopt.json.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_solveopt.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
o.minimizer_indicator = -1; o.TolX = 1e-6; o.TolFun = 1e-6; o.MaxIter = 15000; o.verbosity = 0;
o.TolXConstraint = 1e-8; o.SpaceDilation = 2.5; o.LBGradientStep = 1e-11;
out = struct();
cases = {'rosen2', @rosen5, [-1.2; 1], {10};
         'rosen4', @rosen5, [-1.2; 1; -1.2; 1], {10};
         'quad3', @quad5, [0; 0; 0], {[2 1 0; 0 1 1; 1 0 3; 1 1 1], [1; 2; 3; 4]};
         'rosen2_far', @rosen5, [3; -2], {1}};
for c = 1:size(cases, 1)
    [x, f, ef, nf, ng, ~, ~, k, msg] = solvopt(cases{c,3}, cases{c,2}, [], [], [], o, cases{c,4}{:});
    out.(cases{c,1}) = struct('x', x, 'f', f, 'exitflag', ef, 'n_f_evals', nf, 'n_grad_evals', ng, 'k', k, 'message', msg);
end
% MaxIter override (optim = ('MaxIter', 25)) to compare an early iterate
o2 = o; o2.MaxIter = 25;
[x, f, ef, nf, ng, ~, ~, k, msg] = solvopt([-1.2; 1], @rosen5, [], [], [], o2, 10);
out.rosen2_it25 = struct('x', x, 'f', f, 'exitflag', ef, 'n_f_evals', nf, 'n_grad_evals', ng, 'k', k, 'message', msg);
fid = fopen('ref_solveopt.json', 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
