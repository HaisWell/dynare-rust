% dump_cmaes.m -- statistics of Dynare's cmaes.m (with the mode_compute=9 options) on
% Rosenbrock 5d, box [-5,5], start [-1.2 1 -1.2 1 -1.2], H0 = 0.2*(ub-lb), 20 seeds:
% iterations, evaluations, best fval and stop flags, written to ref_cmaes.json.
% CMA-ES is stochastic: tests/estimation_cmaes.rs compares these distributions only.
% Run from a private copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_cmaes.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
o = struct('SaveVariables','off','DispFinal','off','WarnOnEqualFunctionValues','no', ...
           'DispModulo','0','LogModulo','0','LogTime','0','TolFun',1e-7,'TolX',1e-7, ...
           'Resume',0,'ReadSignals','off');
n = 5; x0 = [-1.2; 1; -1.2; 1; -1.2];
lb = -5*ones(n,1); ub = 5*ones(n,1);
H0 = (ub-lb)*0.2;
o.LBounds = lb; o.UBounds = ub;
fid = fopen('rosen_plain.m','w');
fprintf(fid, 'function f = rosen_plain(x)\nf = sum(100*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);\n');
fclose(fid);
warning('off','all');
R = struct('it',[],'fc',[],'f',[]);
R.flag = {};
for s = 1:20
    randn('state', s); rand('state', s);
    [~, ~, ~, message, output, BESTEVER] = cmaes('rosen_plain', x0, H0, o);
    R.it(end+1) = output.countiter; R.fc(end+1) = output.evals + 1; R.f(end+1) = BESTEVER.f;
    R.flag{end+1} = strjoin(message, ' ');
end
fid = fopen('ref_cmaes.json','w'); fprintf(fid, '%s', jsonencode(R)); fclose(fid);
