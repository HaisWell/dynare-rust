% dump_korder.m -- higher-order perturbation reference values of Dynare 7.2
% (stream K): dynamic derivatives g1/g2/g3 at the steady state, decision rules at
% order 2 (dyn_second_order_solver and k_order_solver) and 3 (k_order_solver),
% simulations with given shocks (simult_ at order 2/3, pruned or not),
% generalized IRFs with given draws, and pruned-state-space theoretical moments.
%
% Run from a PRIVATE copy of reference/korder (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "mdl='rbc'; run('dump_korder.m')"
% writes ref_<mdl>.json.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('mdl', 'var'), mdl = 'rbc'; end
eval(sprintf('dynare %s -DORDER=3 nointeractive', mdl));
global M_ options_ oo_

R = struct();
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names;
R.lead_lag_incidence = M_.lead_lag_incidence;
R.Sigma_e = M_.Sigma_e; R.params = M_.params;
R.ys = oo_.steady_state;
R.k_order_solver_after_preprocessing = options_.k_order_solver;

% ---- dynamic derivatives at the steady state ---------------------------------
ys = oo_.steady_state; exo = oo_.exo_steady_state';
dyn_endo_ss = repmat(ys, 3, 1);
[g1, T_order, T] = feval([M_.fname '.dynamic_g1'], dyn_endo_ss, oo_.exo_steady_state, M_.params, ys, ...
    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
[g2_v, T_order, T] = feval([M_.fname '.dynamic_g2'], dyn_endo_ss, oo_.exo_steady_state, M_.params, ys, T_order, T);
g3_v = feval([M_.fname '.dynamic_g3'], dyn_endo_ss, oo_.exo_steady_state, M_.params, ys, T_order, T);
R.g1 = full(g1);
R.g2 = [double(M_.dynamic_g2_sparse_indices) g2_v(:)];
R.g3 = [double(M_.dynamic_g3_sparse_indices) g3_v(:)];

gh = {'ghx','ghu','ghxx','ghxu','ghuu','ghs2','ghxxx','ghxxu','ghxuu','ghuuu','ghxss','ghuss','g_0','g_1','g_2','g_3'};
pick = @(dr) cell2struct(cellfun(@(f) getfield_or_empty(dr, f), gh, 'UniformOutput', false), gh, 2);

opt0 = options_;
opt0.irf = 0; opt0.nomoments = 1; opt0.noprint = 1; opt0.nograph = 1; opt0.periods = 0;

% ---- order 2, dyn_second_order_solver --------------------------------------------
o = opt0; o.order = 2; o.k_order_solver = 0;
[info, oo2, o2] = stoch_simul(M_, o, oo_, {});
R.o2 = pick(oo2.dr); R.order_var = oo2.dr.order_var; R.state_var = oo2.dr.state_var;
R.nstatic = M_.nstatic; R.npred = M_.npred; R.nboth = M_.nboth; R.nfwrd = M_.nfwrd;
% ---- order 2, k_order_solver -------------------------------------------------------
o = opt0; o.order = 2; o.k_order_solver = 1;
[info, oo2k] = stoch_simul(M_, o, oo_, {});
R.o2k = pick(oo2k.dr);
% ---- order 3, k_order_solver -------------------------------------------------------
o = opt0; o.order = 3; o.k_order_solver = 1;
[info, oo3k, o3] = stoch_simul(M_, o, oo_, {});
R.o3k = pick(oo3k.dr);

% ---- simulations with given shocks ------------------------------------------------
rng(42);
T = 60;
ex = randn(T, M_.exo_nbr) * chol(M_.Sigma_e) * 3;  % large shocks: nonlinear terms matter
R.sim_ex = ex;
y0 = oo_.steady_state;
y0b = y0 + 0.02*abs(y0) .* (1:M_.endo_nbr)'/M_.endo_nbr;  % and off it
R.sim_y0b = y0b;
o = opt0; o.order = 2; o.pruning = false;
R.sim_o2 = simult_(M_, o, y0b, oo2.dr, ex, 2);
o.pruning = true;
R.sim_o2_pruned = simult_(M_, o, y0b, oo2.dr, ex, 2);
o = opt0; o.order = 3; o.k_order_solver = 1; o.pruning = true;
R.sim_o3_pruned = simult_(M_, o, y0b, oo3k.dr, ex, 3);
o.pruning = false;
R.sim_o3 = simult_(M_, o, y0b, oo3k.dr, ex, 3);

% ---- GIRF with given draws (irf.m, order 2 and 3) -------------------------------
long = 12; drop = 5; replic = 3;
i_exo_var = setdiff(1:M_.exo_nbr, find(diag(M_.Sigma_e) == 0));
chol_S = chol(M_.Sigma_e(i_exo_var, i_exo_var));
draws = cell(replic, 1);
for j = 1:replic
    draws{j} = randn(long+drop, numel(i_exo_var)) * chol_S;
end
R.girf_draws = draws; R.girf_long = long; R.girf_drop = drop;
cs = get_lower_cholesky_covariance(M_.Sigma_e, opt0.add_tiny_number_to_cholesky);
R.girf_cs = cs;
temps = repmat(oo_.steady_state, 1, M_.maximum_lag);
for ord = [2 3]
    if ord == 2, dr = oo2.dr; else, dr = oo3k.dr; end
    for pr = [0 1]
        o = opt0; o.order = ord; o.pruning = logical(pr); o.k_order_solver = (ord == 3);
        y = 0;
        e1 = cs(:, 1);
        for j = 1:replic
            ex1 = zeros(long+drop, M_.exo_nbr);
            ex1(:, i_exo_var) = draws{j};
            ex2 = ex1; ex2(drop+1, :) = ex2(drop+1, :) + e1';
            y1 = simult_(M_, o, temps, dr, ex1, ord);
            y2 = simult_(M_, o, temps, dr, ex2, ord);
            y = y + (y2(:, M_.maximum_lag+drop+1:end) - y1(:, M_.maximum_lag+drop+1:end));
        end
        R.(sprintf('girf_o%d_p%d', ord, pr)) = y / replic;
    end
end

% ---- theoretical moments -------------------------------------------------------
o = opt0; o.order = 2; o.k_order_solver = 0; o.pruning = false; o.nomoments = 0; o.ar = 5;
[info, oom] = stoch_simul(M_, o, oo_, {});
R.mom_o2 = struct('mean', oom.mean, 'var', oom.var, 'autocorr', {oom.autocorr});
o.pruning = true;
[info, oom] = stoch_simul(M_, o, oo_, {});
R.mom_o2_pruned = struct('mean', oom.mean, 'var', oom.var, 'autocorr', {oom.autocorr});
o = opt0; o.order = 3; o.k_order_solver = 1; o.pruning = true; o.nomoments = 0; o.ar = 5;
[info, oom] = stoch_simul(M_, o, oo_, {});
R.mom_o3_pruned = struct('mean', oom.mean, 'var', oom.var, 'autocorr', {oom.autocorr});
% full pruned state-space objects (order 2 and 3)
o = opt0; o.order = 2; o.pruning = true;
pss2 = pruned_SS.pruned_state_space_system(M_, o, oo2.dr, (1:M_.endo_nbr)', 5, false, false);
o.order = 3;
pss3 = pruned_SS.pruned_state_space_system(M_, o, oo3k.dr, (1:M_.endo_nbr)', 5, false, false);
R.pss2 = struct('E_y', pss2.E_y, 'Var_y', pss2.Var_y);
R.pss3 = struct('E_y', pss3.E_y, 'Var_y', pss3.Var_y);

fid = fopen(sprintf('ref_%s.json', mdl), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
disp('dump_korder: done');

function v = getfield_or_empty(s, f)
if isfield(s, f), v = s.(f); else, v = []; end
end
