% dump_simplex.m -- reference values for tests/estimation_simplex.rs: Dynare's
% simplex_optimization_routine.m (mode_compute=8) and MATLAB's fminsearch (mode_compute=7)
% with Dynare's options, on rosen5/quad5. Writes ref_simplex.json.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_simplex.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
simplex.tolerance.x = 1e-4; simplex.tolerance.f = 1e-4; simplex.maxiter = 10000;
simplex.maxfcallfactor = 500; simplex.maxfcall = []; simplex.verbosity = 0; simplex.delta_factor = 0.05;
out = struct();
cases = {{'r2', @rosen5, [-1.2; 1], {10}}, {'r4', @rosen5, [-1.2; 1; -1.2; 1], {10}}, ...
         {'q3', @quad5, [0.3; -0.2; 0.5], {[2 1 0; 0 1 1; 1 0 3; 1 1 1], [1; 2; 3; 4]}}, ...
         {'r5', @rosen5, [0; 0; 0; 0; 0], {10}}};
for k = 1:numel(cases)
    c = cases{k};
    [x, f, ef, it, fc, msg] = simplex_optimization_routine(c{2}, c{3}, simplex, {}, c{4}{:});
    out.(['s_' c{1}]) = struct('x', x, 'f', f, 'ef', ef, 'it', it, 'fc', fc, 'msg', msg);
    o = optimset('display','off','MaxFunEvals',1000000,'MaxIter',6000,'TolFun',1e-8,'TolX',1e-6);
    [x, f, ef, output] = fminsearch(c{2}, c{3}, o, c{4}{:});
    out.(['f_' c{1}]) = struct('x', x, 'f', f, 'ef', ef, 'it', output.iterations, 'fc', output.funcCount);
end
% maxfcall cap
s2 = simplex; s2.maxfcall = 300;
[x, f, ef, it, fc, msg] = simplex_optimization_routine(@rosen5, [-1.2; 1; -1.2; 1], s2, {}, 10);
out.s_cap = struct('x', x, 'f', f, 'ef', ef, 'it', it, 'fc', fc, 'msg', msg);
fid = fopen('ref_simplex.json','w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
