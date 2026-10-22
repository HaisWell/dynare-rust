% dump_stoch_simul.m -- stoch_simul (order 1) and check of Dynare on the SW07 model at
% the posterior mode, written to ref_stoch_simul.json (reference for src/analysis).
% SW07 has unit roots: the theoretical moments keep only the stationary variables (NaN
% elsewhere), as th_autocovariances.m decides from the unit-root Schur vectors.
%
% Run from a PRIVATE copy of reference/sw07 (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_stoch_simul.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_

mode = load('sw07/Output/sw07_mode.mat');
xm = mode.xparam1(:);
M_ = set_all_parameters(xm, estim_params_, M_);

R = struct();
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.xparam_mode = xm;
R.estim_param_names = mode.parameter_names; R.Sigma_e = M_.Sigma_e;

% check at the mode
options_.noprint = 0;
[eigenvalues_, result, info] = check(M_, options_, oo_);
R.check = struct('eigval_re', real(eigenvalues_), 'eigval_im', imag(eigenvalues_), 'result', result, 'info', info(1));

% stoch_simul on every declared variable: IRFs, theoretical moments, decompositions
options_.order = 1; options_.irf = 12; options_.ar = 3; options_.nograph = 1; options_.periods = 0;
options_.hp_filter = 0; options_.conditional_variance_decomposition = [1 4 12]; options_.noprint = 1;
var_list = M_.endo_names(1:M_.orig_endo_nbr);
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list);
C = struct();
C.info = info(1);
C.var_list = var_list;
C.ghx = oo_.dr.ghx; C.ghu = oo_.dr.ghu; C.order_var = oo_.dr.order_var; C.ys = oo_.dr.ys;
C.irfs = oo_.irfs; C.gamma_y = oo_.gamma_y; C.mean = oo_.mean; C.var = oo_.var;
C.autocorr = oo_.autocorr; C.variance_decomposition = oo_.variance_decomposition;
C.conditional_variance_decomposition = oo_.conditional_variance_decomposition;
C.cvd_steps = options_.conditional_variance_decomposition;
R.theoretical = C;

% HP-filtered theoretical moments on the observables
oo_ = rmfield(oo_, 'irfs');
options_.hp_filter = 1600; options_.ar = 2; options_.irf = 0; options_.conditional_variance_decomposition = [];
var_list = options_.varobs(:);
[info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list);
H = struct();
H.var_list = var_list; H.gamma_y = oo_.gamma_y; H.var = oo_.var; H.variance_decomposition = oo_.variance_decomposition;
R.hp = H;

fid = fopen('ref_stoch_simul.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_stoch_simul.json');
