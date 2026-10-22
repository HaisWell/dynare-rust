% dump_q1xdet.m -- varexo_det reference (q1_xdet.mod): decision rules (ghx, ghu,
% ghud), oo_.exo_det_simul, forecasts with the deterministic path, and the printed output
% (q1xdet.log). Run from a PRIVATE copy of reference/q1:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q1xdet.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
global M_ options_ oo_
txt = evalc('dynare q1_xdet nointeractive nograph noclearall');
fid = fopen('q1xdet.log', 'w'); fprintf(fid, '%s', txt); fclose(fid);
R = struct();
R.endo_names = M_.endo_names;
R.exo_names = M_.exo_names;
R.exo_det_names = M_.exo_det_names;
R.ys = oo_.dr.ys;
R.ghx = oo_.dr.ghx; R.ghu = oo_.dr.ghu; R.order_var = oo_.dr.order_var;
R.ghud = oo_.dr.ghud;
R.exo_det_length = oo_.dr.exo_det_length;
R.exo_det_simul = oo_.exo_det_simul;
R.exo_det_steady_state = oo_.exo_det_steady_state;
R.forecast = oo_.forecast;
fid = fopen('ref_q1xdet.json', 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
