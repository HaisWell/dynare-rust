% run_ref.m -- reference values for tests/estimation_*.rs (csminwel1/newrat iterates,
% hessian.m, priors of every shape, prior_trunc bounds), written to ml_ref.json.
% Run from a private copy of this folder (the SW07 prior_trunc bounds and chi2inv are in
% ../sw07/dump_prior_trunc.m):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run_ref"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
out = struct();
% csminwel on Rosenbrock (2d and 4d)
x0 = [-1.2; 1];
[fh,xh,gh,H,itct,fcount,retcodeh] = csminwel1(@rosen5, x0, 1e-4*eye(2), [], 1e-7, 1000, 2, 1e-6, 0, 0, 10);
out.cw2 = struct('f',fh,'x',xh,'it',itct,'fc',fcount,'rc',retcodeh,'H',H);
x0 = [-1.2; 1; -1.2; 1];
[fh,xh,gh,H,itct,fcount,retcodeh] = csminwel1(@rosen5, x0, 1e-4*eye(4), [], 1e-7, 1000, 2, 1e-6, 0, 0, 10);
out.cw4 = struct('f',fh,'x',xh,'it',itct,'fc',fcount,'rc',retcodeh,'H',H);
% newrat on Rosenbrock 2d with bounds
hess_info.gstep = [1e-2; 1]; hess_info.htol = 1e-5; hess_info.htol_rel = NaN;
hess_info.h1 = 1e-6*ones(2,1); hess_info.robust = false;
bounds = [-5 5; -5 5];
[xp, hh, gg, fval, igg, hi, jit, fcount, exitflag] = newrat(@rosen5, [-1.2; 1], bounds, 0, 1e-5, 1000, 0, 0, 0, hess_info, NaN(2,1), 1e-6, {'a','b'}, 10);
out.nr2 = struct('f',fval,'x',xp,'it',jit,'fc',fcount,'ef',exitflag,'hh',hh);
% newrat on a quadratic in 3d
A = [2 1 0; 0 1 1; 1 0 3; 1 1 1]; b = [1; 2; 3; 4];
hess_info.h1 = 1e-6*ones(3,1);
bounds = [-10 10; -10 10; -10 10];
[xp, hh, gg, fval, igg, hi, jit, fcount, exitflag] = newrat(@quad5, [0;0;0], bounds, 0, 1e-5, 1000, 0, 0, 0, hess_info, NaN(3,1), 1e-6, {'a','b','c'}, A, b);
out.nrq = struct('f',fval,'x',xp,'it',jit,'fc',fcount,'ef',exitflag,'hh',hh);
% hessian.m at a point
out.hess = hessian(@rosen5, [0.5; 0.7], [1e-2; 1], 10);
% priors of all shapes (as in priordens.m unit test) and truncated bounds
p0 = [1;2;3;4;5;6;8]; p1 = .4*ones(7,1); p2 = .2*ones(7,1);
p3 = NaN(7,1); p4 = NaN(7,1); p5 = NaN(7,1); p6 = NaN(7,1); p7 = NaN(7,1);
for i=1:7
  switch p0(i)
    case 1, p3(i)=0; p4(i)=1; [p6(i),p7(i)] = beta_specification(p1(i),p2(i)^2,p3(i),p4(i));
    case 2, p3(i)=0; p4(i)=Inf; [p6(i),p7(i)] = gamma_specification(p1(i),p2(i)^2,p3(i));
    case 3, p3(i)=-Inf; p4(i)=Inf; p6(i)=p1(i); p7(i)=p2(i);
    case 4, p3(i)=0; p4(i)=Inf; [p6(i),p7(i)] = inverse_gamma_specification(p1(i),p2(i)^2,p3(i),1,false);
    case 5, [p1(i),p2(i),p6(i),p7(i)] = uniform_specification(p1(i),p2(i),p3(i),p4(i)); p3(i)=p6(i); p4(i)=p7(i);
    case 6, p3(i)=0; p4(i)=Inf; [p6(i),p7(i)] = inverse_gamma_specification(p1(i),p2(i)^2,p3(i),2,false);
    case 8, p3(i)=0; p4(i)=Inf; [p6(i),p7(i)] = weibull_specification(p1(i),p2(i)^2,p3(i));
  end
  p5(i) = compute_prior_mode([p6(i) p7(i) p3(i) p4(i)], p0(i));
end
bt.pshape=p0; bt.p1=p1; bt.p2=p2; bt.p3=p3; bt.p4=p4; bt.p5=p5; bt.p6=p6; bt.p7=p7;
out.shapes = struct('p1',p1,'p2',p2,'p3',p3,'p4',p4,'p5',p5,'p6',p6,'p7',p7);
bb = prior_bounds(bt, 1e-10); out.shapes_trunc = struct('lb',bb.lb,'ub',bb.ub);
xs = [0.3; 0.5; -0.1; 0.35; 0.2; 0.45; 0.6];
out.shapes_lp = priordens(xs, p0, p6, p7, p3, p4);
fid = fopen('ml_ref.json','w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
