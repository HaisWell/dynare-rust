% check_sa_simpsa.m -- behaviour of Dynare's simulated_annealing.m (mode_compute=2) and
% simpsa.m (mode_compute=10) on the test functions of tests/estimation_sa_simpsa.rs, with
% Dynare's default options. Both are stochastic: compare magnitudes, not digits.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('check_sa_simpsa.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab/optimization'));
addpath(fullfile(droot, 'matlab/utilities/general'));
addpath(fullfile(droot, 'matlab'));
rosen = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);
quad = @(x) 3 + sum([1;10;0.5;4].*(x(:)-[1.5;-2;0.5;3]).^2);
o.neps=10; o.maximizer_indicator=0; o.rt=0.10; o.verbosity=0; o.TolFun=1e-8;
o.initial_temperature=15; o.ns=10; o.nt=10;
for n=[2 5]
    x0 = -1.2*ones(n,1); x0(end) = 1;
    o.step_length_c=0.1*ones(n,1); o.initial_step_length=ones(n,1);
    for maxit=[1e5 5e6]
        o.MaxIter=maxit;
        for s=1:3
            rng(s);
            [x,f,ef,na,nt]=simulated_annealing(rosen,x0,o,-5*ones(n,1),5*ones(n,1));
            fprintf('SA rosen%d MaxIter %g seed %d: f %g evals %d exit %d\n', n, maxit, s, f, nt, ef);
        end
    end
end
o.step_length_c=0.1*ones(4,1); o.initial_step_length=ones(4,1); o.MaxIter=1e5;
rng(1); [x,f,ef,na,nt]=simulated_annealing(quad,zeros(4,1),o,-10*ones(4,1),10*ones(4,1));
fprintf('SA quad: f %.15g evals %d exit %d\n', f, nt, ef);
% simpsa with options_.simpsa
so = simpsaset('TOLFUN',1e-4,'TOLX',1e-4,'TEMP_END',.1,'COOL_RATE',10,'INITIAL_ACCEPTANCE_RATIO',.95, ...
    'MIN_COOLING_FACTOR',.9,'MAX_ITER_TEMP_FIRST',50,'MAX_ITER_TEMP_LAST',2000,'MAX_ITER_TEMP',10, ...
    'MAX_ITER_TOTAL',5000,'MAX_TIME',2500,'MAX_FUN_EVALS',20000,'DISPLAY','none');
for n=[2 5]
    x0 = -1.2*ones(n,1); x0(end) = 1;
    for s=1:3
        rng(s);
        [x,f,ef,out]=simpsa('simpsa_rosen',x0,-5*ones(n,1),5*ones(n,1),so);
        fprintf('SIMPSA rosen%d seed %d: f %g evals %d iter %d exit %d\n', n, s, f, out.nFUN_EVALS, out.nITERATIONS, ef);
    end
end
for s=1:3
    rng(s);
    [x,f,ef,out]=simpsa('simpsa_quad',zeros(4,1),-10*ones(4,1),10*ones(4,1),so);
    fprintf('SIMPSA quad seed %d: f %.10g evals %d iter %d exit %d\n', s, f, out.nFUN_EVALS, out.nITERATIONS, ef);
end
