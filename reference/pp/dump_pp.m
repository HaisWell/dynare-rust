% dump_pp.m -- numerical reference values of Dynare (7.2 by default, see DYNARE_VERSION) for the stream PP test models:
% dynamic residuals and Jacobian at a perturbed point (with the steady state passed to the
% STEADY_STATE() operator), static residuals/Jacobian, the steady state, the first-order
% decision rules (stochastic models) and the perfect-foresight paths (deterministic model).
%
% Run from a PRIVATE copy of reference/pp (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "mdl='pp_ss'; run('dump_pp.m')"
% writes ref_<mdl>.json.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('mdl', 'var'), mdl = 'pp_ss'; end
global M_ options_ oo_
try
    eval(sprintf('dynare %s nointeractive nograph', mdl));
catch err
    % e.g. Blanchard-Kahn failures of the toy models: the model files are still there
    disp(getReport(err));
end

R = struct();
R.endo_names = M_.endo_names; R.params = M_.params;
n = M_.endo_nbr; nx = M_.exo_nbr;
ys = oo_.steady_state;
if any(~isfinite(ys)) || all(ys == 0)
    ys = oo_.steady_state;
end
R.ys = ys;
% perturbed point: y_{t-1}, y_t, y_{t+1} around ys, generic exogenous values
k = (1:n)';
y3 = [ys .* (1 + 0.010 * k / n) + 0.001; ys .* (1 - 0.005 * k / n) + 0.002; ys .* (1 + 0.007 * k / n) - 0.001];
x = 0.01 * (1:nx)' + 0.003;
R.y3 = y3; R.x = x;
[r, T_order, T] = feval([M_.fname '.dynamic_resid'], y3, x, M_.params, ys);
R.dynamic_resid = r;
g1 = feval([M_.fname '.dynamic_g1'], y3, x, M_.params, ys, ...
    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
R.dynamic_g1 = full(g1);
ystat = ys .* (1 + 0.004 * k / n) + 0.001;
R.ystat = ystat;
R.static_resid = feval([M_.fname '.static_resid'], ystat, x, M_.params);
R.static_g1 = full(feval([M_.fname '.static_g1'], ystat, x, M_.params, ...
    M_.static_g1_sparse_rowval, M_.static_g1_sparse_colval, M_.static_g1_sparse_colptr));
if isfield(oo_, 'dr') && isfield(oo_.dr, 'ghx')
    R.ghx = oo_.dr.ghx; R.ghu = oo_.dr.ghu; R.order_var = oo_.dr.order_var;
    if isfield(oo_.dr, 'ghxx')
        R.ghxx = oo_.dr.ghxx; R.ghxu = oo_.dr.ghxu; R.ghuu = oo_.dr.ghuu; R.ghs2 = oo_.dr.ghs2;
    end
end
if isfield(oo_, 'endo_simul') && size(oo_.endo_simul, 2) > 1 && options_.periods > 0
    R.endo_simul = oo_.endo_simul;
end
fid = fopen(sprintf('ref_%s.json', mdl), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
