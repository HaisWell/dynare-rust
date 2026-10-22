% Run from a private copy of reference/varmodels, one fresh MATLAB per model (M_.var_expectation
% survives `noclearall` runs):
%   for m in vm1 vm2 vm3; do VM_NAME=$m /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump.m')" > log_$m.txt 2>&1; done
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
name = getenv('VM_NAME');
dynare(name, 'nolog');
r = struct();
r.param_names = M_.param_names;
r.params = M_.params;
r.endo_names = M_.endo_names;
if isfield(oo_, 'var'), r.var = oo_.var; end
if isfield(oo_, 'trend_component'), r.trend_component = oo_.trend_component; end
if isfield(oo_, 'dr') && isfield(oo_.dr, 'ghx')
    r.ys = oo_.dr.ys;
    r.ghx = oo_.dr.ghx;
    r.ghu = oo_.dr.ghu;
    r.order_var = oo_.dr.order_var;
    r.irfs = oo_.irfs;
end
fid = fopen(['ref_' name '.json'], 'w'); fprintf(fid, '%s', jsonencode(r)); fclose(fid);
