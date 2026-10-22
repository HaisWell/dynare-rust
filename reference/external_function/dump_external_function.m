% dump_external_function.m -- run Dynare (7.2 by default, see DYNARE_VERSION) on ext_rbc.mod with the three derivative
% settings of external_function (-DDERIVS=0,1,2) and write the steady state and the order-2
% decision rules to ref_ext_rbc_d<k>.json (reference for tests/external_function.rs).
%
% Run from a PRIVATE copy of reference/external_function (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_external_function.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
for d = 0:2
    eval(sprintf('dynare ext_rbc -DDERIVS=%d nointeractive', d));
    R = struct();
    R.endo_names = M_.endo_names;
    R.steady_state = oo_.steady_state;
    R.order_var = oo_.dr.order_var;
    R.ghx = oo_.dr.ghx; R.ghu = oo_.dr.ghu;
    R.ghxx = oo_.dr.ghxx; R.ghxu = oo_.dr.ghxu; R.ghuu = oo_.dr.ghuu; R.ghs2 = oo_.dr.ghs2;
    fid = fopen(sprintf('ref_ext_rbc_d%d.json', d), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
    fclose(fid);
end
disp('wrote ref_ext_rbc_d*.json');
