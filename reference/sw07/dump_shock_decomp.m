% dump_shock_decomp.m -- reference values for the Rust shock decomposition
% (ref_shock_decomp.json).
%
% STAGE 9 run (mode file, smoother), then shock_decomposition.m at the posterior mode on
% the estimation's sample: oo_.shock_decomposition (endo_nbr x (exo_nbr+2) x nobs; column
% exo_nbr+1 = initial conditions, exo_nbr+2 = smoothed deviation from the steady state).
% Also the same with init_state = 2 (shocks only contribute after period 2).
%
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_shock_decomp.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

options_.parameter_set = 'posterior_mode';
options_.no_graph.shock_decomposition = true;
oo0 = oo_;
oo_ = shock_decomposition(M_, oo_, options_, var_list_, bayestopt_, estim_params_);
z = oo_.shock_decomposition;

options_.shock_decomp.init_state = 2;
oo2 = shock_decomposition(M_, oo0, options_, var_list_, bayestopt_, estim_params_);
z2 = oo2.shock_decomposition;

R = struct();
R.endo_names = M_.endo_names(1:M_.orig_endo_nbr);
R.exo_names = M_.exo_names;
R.var_list = var_list_;
% orig endo only: [orig_endo_nbr, exo_nbr+2, nobs], flattened in MATLAB column-major order
R.size = size(z(1:M_.orig_endo_nbr,:,:));
R.z = reshape(z(1:M_.orig_endo_nbr,:,:), [], 1);
R.z_init_state2 = reshape(z2(1:M_.orig_endo_nbr,:,:), [], 1);

fid = fopen('ref_shock_decomp.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_shock_decomp.json written\n');
