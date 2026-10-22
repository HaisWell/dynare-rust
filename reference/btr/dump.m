% dump.m -- block_trust_region (solve_algo 13/14) reference for src/solver/block_trust_region.rs.
% Run from a PRIVATE copy of reference/btr:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log.txt 2>&1
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']); addpath([droot '/mex/matlab/maca64-23.2-26.1']);
here = fileparts(mfilename('fullpath')); cd(here);
tolf = eps^(1/3); tolx = eps^(2/3);
R = struct();
% direct MEX cases: {name, handle, x0, maxit, factor, block_decompose}
C = {
 'rosen',   @f_rosen,  [-1.2;1],            50, 1,   false
 'rosen_b', @f_rosen,  [-1.2;1],            50, 1,   true
 'chain',   @f_chain,  [3;1;2;0.5],         50, 1,   true
 'chain_w', @f_chain,  [3;1;2;0.5],         50, 1,   false
 'chain_f', @f_chain,  [30;10;5;5],         50, 100, true
 'maxit',   @f_exp,    [8;-6],               4, 1,   false
 'singular',@f_sing,   [0;0.5],             50, 1,   false
 'sing_dm', @f_sing,   [0;0.5],             50, 1,   true
 'sol0',    @f_rosen,  [1;1],               50, 1,   true
 'exp50',   @f_exp,    [8;-6],              50, 1,   true
};
for i = 1:size(C,1)
    [x, ef, ec] = block_trust_region(C{i,2}, C{i,3}, tolf, tolx, C{i,4}, C{i,5}, C{i,6}, false);
    R.(C{i,1}) = struct('x', x(:)', 'errorflag', ef, 'errorcode', ec);
end
% steady states through dynare
for a = [13 14]
    try
        txt = strrep(fileread('btr_ss.mod'), 'solve_algo=13', sprintf('solve_algo=%d', a));
        fid0 = fopen('btr_tmp.mod','w'); fprintf(fid0, '%s', txt); fclose(fid0);
        evalc('dynare btr_tmp.mod nointeractive nograph noclearall');
        global oo_ M_
        R.(sprintf('ss%d', a)) = struct('ok', true, 'x', oo_.steady_state(:)');
    catch err
        R.(sprintf('ss%d', a)) = struct('ok', false, 'x', [], 'msg', err.message);
        disp(err.message)
    end
end
fid = fopen('ref_btr.json','w'); fprintf(fid, '%s', jsonencode(R)); fclose(fid);
exit
function [f,J] = f_rosen(x)
f = [10*(x(2)-x(1)^2); 1-x(1)];
J = [-20*x(1) 10; -1 0];
end
function [f,J] = f_chain(x)
f = [x(1)^3-8; x(2)+x(1)*x(2)^2-3; x(3)^2+x(4)^2-5; x(3)*x(4)-2+0.1*x(2)];
J = [3*x(1)^2 0 0 0; x(2)^2 1+2*x(1)*x(2) 0 0; 0 0 2*x(3) 2*x(4); 0 0.1 x(4) x(3)];
end
function [f,J] = f_exp(x)
f = [exp(x(1))-2*x(2)-1; x(1)+exp(x(2))-2];
J = [exp(x(1)) -2; 1 exp(x(2))];
end
function [f,J] = f_sing(x)
f = [x(1)^2; x(2)-1];
J = [2*x(1) 0; 0 1];
end
