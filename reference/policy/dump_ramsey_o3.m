% Reference for evaluate_planner_objective at order 3 (k_order_welfare + k_order_mean, with the
% `periods`/`drop` options): writes ref_ramsey_cmr_o3.json.
% Run from a private copy of reference/policy:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_ramsey_o3.m')"
% The shocks of the simulated unconditional welfare are drawn with randn after rng(1, 'twister');
% they are redrawn here (same seed) and stored, so the Rust test can feed the same shocks.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
clearvars -global
global M_ oo_ options_
dynare ramsey_cmr_o3 noclearall nolog
dr = oo_.dr;
r = struct();
r.endo_names = M_.endo_names;
r.order_var = dr.order_var;
r.ys = dr.ys;
r.nstatic = M_.nstatic; r.npred = M_.npred; r.nboth = M_.nboth; r.nfwrd = M_.nfwrd;
for i = 0:3
    r.(['g_' num2str(i)]) = dr.(['g_' num2str(i)]);
end
W = k_order_welfare(dr, M_, options_);
for i = 0:3
    r.(['W_' num2str(i)]) = W.(['W_' num2str(i)]);
end
r.ramsey_periods = options_.ramsey.periods;
r.ramsey_drop = options_.ramsey.drop;
% evaluate_planner_objective with known shocks
opts = options_; opts.noprint = 1;
rng(1, 'twister');
v = evaluate_planner_objective(M_, opts, oo_);
r.planner_objective_value = v;
i_exo_var = setdiff(1:M_.exo_nbr, find(diag(M_.Sigma_e) == 0));
rng(1, 'twister');
ex = zeros(M_.exo_nbr, opts.ramsey.periods);
ex(i_exo_var, :) = chol(M_.Sigma_e(i_exo_var, i_exo_var))*randn(numel(i_exo_var), opts.ramsey.periods);
r.exo_simul = ex;
% k_order_mean semantics (first period skipped, divided by periods - drop)
dr2 = dr;
for i = 0:3
    dr2.(['g_' num2str(i)]) = [dr.(['g_' num2str(i)]); W.(['W_' num2str(i)])];
end
U = feval([M_.fname '.objective.static_resid'], dr.ys, zeros(1, M_.exo_nbr), M_.params);
beta = get_optimal_policy_discount_factor(M_.params, M_.param_names);
ysteady = [dr.ys(dr.order_var); U/(1-beta)];
r.probe_ex = [0.01 -0.02 0.005 0.0];
y0 = zeros(M_.endo_nbr+1, 1);
y0(M_.nstatic+1) = 0.05;
r.probe_ystart = y0;
for d = 0:3
    [m, s] = k_order_mean(3, M_.nstatic, M_.npred, M_.nboth, M_.nfwrd+1, M_.exo_nbr, 1, d, y0, r.probe_ex, ysteady, dr2);
    r.(['probe_mean_drop' num2str(d)]) = m;
end
r.probe_sim = s;
fid = fopen('ref_ramsey_cmr_o3.json', 'w');
fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', true));
fclose(fid);
