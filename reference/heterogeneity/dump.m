% dump.m -- reference values of Dynare (DYNARE_VERSION, default 7.2) for the heterogeneous-agent models of this folder.
% Run from a PRIVATE copy of reference/heterogeneity (Dynare writes next to the .mod), one
% fresh MATLAB session per model:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "mdl='ks'; run('dump.m')"  > log_ks.txt
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "mdl='ksl'; run('dump.m')" > log_ksl.txt
% writes ref_<mdl>.json (and ks_guess.mat for ks, the initial guess of the verbatim block of
% examples/heterogeneity/krusell_smith_1998_steady_state.mod).
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
dynare_config;
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('mdl', 'var'), mdl = 'ks'; end
global M_ options_ oo_

if strcmp(mdl, 'ks')
    delta = 0.025; eis = 1; L = 1; alpha = 0.11; beta = 0.98; %#ok<NASGU>
    r = 0.01;
    rk = r + delta;
    Z_ss = (rk / alpha) ^ alpha;
    K = (alpha * Z_ss / rk) ^ (1 / (1 - alpha));
    Y = Z_ss * K ^ alpha;
    w = (1 - alpha) * Z_ss * (alpha * Z_ss / rk) ^ (alpha / (1 - alpha));
    initial_guess = struct;
    initial_guess.agg.w = w;
    initial_guess.agg.r = r;
    initial_guess.agg.Y = Y;
    initial_guess.agg.K = K;
    initial_guess.free_parameters.beta.initial_guess = 0.98;
    initial_guess.free_parameters.beta.lower_bound = 0.97;
    initial_guess.free_parameters.beta.upper_bound = 0.99;
    rho_e = 0.966;
    sig_e = 0.5;
    [grid_e, ~, Pi_e] = rouwenhorst(rho_e, sig_e, 3, 1e-12, 1e5);
    initial_guess.shocks.grids.e = grid_e;
    initial_guess.shocks.Pi.e = Pi_e;
    grid_a = logspace(log10(0.25), log10(200.25), 30)-0.25;
    initial_guess.pol.grids.a = grid_a;
    coh = (1+r)*grid_a + w*grid_e;
    c = 0.1*coh;
    a = coh-c;
    initial_guess.pol.values.c = c;
    initial_guess.pol.values.a = a;
    initial_guess.pol.order = {'e', 'a'};
    save('ks_guess.mat', 'initial_guess');
end

eval(sprintf('dynare %s nointeractive nograph', mdl));
% models whose verbatim block builds initial_guess (hank1 = examples/heterogeneity/
% hank_one_asset_steady_state.mod): saved for the Rust test
if exist('initial_guess', 'var') && ~strcmp(mdl, 'ks')
    save(sprintf('%s_guess.mat', mdl), 'initial_guess');
end

flat = @(x) reshape(x, 1, []);
R = struct();
R.endo_names = M_.endo_names; R.orig_endo_nbr = M_.orig_endo_nbr;
R.exo_names = M_.exo_names; R.param_names = M_.param_names; R.params = M_.params;
R.lead_lag_incidence = M_.lead_lag_incidence;
R.aux_vars = M_.aux_vars;
R.heterogeneity_aggregates = M_.heterogeneity_aggregates;
R.equations_tags = M_.equations_tags;
H = M_.heterogeneity(1);
R.het = H;
R.agg_rowval = M_.dynamic_g1_sparse_rowval; R.agg_colval = M_.dynamic_g1_sparse_colval;

% ---- numerical evaluation of the generated code at a fixed point
nh = H.endo_nbr; nhx = H.exo_nbr; na = M_.endo_nbr; nx = M_.exo_nbr;
rng(1);
yh = 0.5 + rand(3*nh, 1); xh = 0.3 + rand(nhx, 1);
y = 0.5 + rand(3*na, 1); x = 0.1*rand(nx, 1);
yagg = 0.7 + rand(numel(H.dynamic_g1_sparse_rowval) * 0 + size(M_.heterogeneity_aggregates, 1), 1);
R.eval.yh = yh; R.eval.xh = xh; R.eval.y = y; R.eval.x = x; R.eval.yagg = yagg;
R.eval.het_resid = feval([M_.fname '.dynamic_het1_resid'], y, x, M_.params, [], yh, xh, []);
g1 = feval([M_.fname '.dynamic_het1_g1'], y, x, M_.params, [], yh, xh, [], ...
    H.dynamic_g1_sparse_rowval, H.dynamic_g1_sparse_colval, H.dynamic_g1_sparse_colptr);
R.eval.het_g1 = full(g1);
for step = 0:H.n_aux_levels-1
    yh2 = feval([M_.fname '.dynamic_het1_set_auxiliary_variables'], y, x, M_.params, [], yh, xh, [], step);
    R.eval.(sprintf('het_aux_step%d', step)) = yh2;
end
[lb, ub] = feval([M_.fname '.dynamic_het1_complementarity_conditions'], M_.params);
R.eval.lb = lb; R.eval.ub = ub;
R.eval.agg_resid = feval([M_.fname '.dynamic_resid'], y, x, M_.params, [], yagg);
g = feval([M_.fname '.dynamic_g1'], y, x, M_.params, [], yagg, M_.dynamic_g1_sparse_rowval, ...
    M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
R.eval.agg_g1 = full(g);
ys = (1:na)'/na + 0.3;
R.eval.set_aux_in = ys;
R.eval.set_aux_out = feval([M_.fname '.set_auxiliary_variables'], ys, zeros(nx, 1), yagg, M_.params);

% ---- the steady state in oo_.heterogeneity
S = oo_.heterogeneity;
ss = S.steady_state;
R.ss.agg = ss.agg;
R.ss.pol_order = ss.pol.order;
fn = fieldnames(ss.pol.values);
for i = 1:numel(fn)
    R.ss.pol_values.(fn{i}) = struct('size', size(ss.pol.values.(fn{i})), 'data', flat(ss.pol.values.(fn{i})));
end
R.ss.pol_grids = ss.pol.grids; R.ss.d_grids = ss.d.grids;
R.ss.d_hist = struct('size', size(ss.d.hist), 'data', flat(ss.d.hist));
R.ss.shocks = ss.shocks;
R.sizes = S.sizes;
R.indices.states = S.indices.states; R.indices.shocks = S.indices.shocks; R.indices.order = S.indices.order;
R.indices.Ix_in_agg = S.indices.Ix.in_agg; R.indices.Ix_in_het = S.indices.Ix.in_het;
R.indices.mult_in_het = S.indices.mult.in_het;
R.indices.Y_all = S.indices.Y.ind_in_dG.all; R.indices.Y_exo = S.indices.Y.ind_in_dG.exo;
R.indices.Y_lead = S.indices.Y.ind_in_dG.lead; R.indices.Y_current = S.indices.Y.ind_in_dG.current;
R.indices.Y_lag = S.indices.Y.ind_in_dG.lag;
m = S.mat;
R.mat.Mu = m.Mu; R.mat.sm = m.pol.sm; R.mat.x_bar = m.pol.x_bar;
R.mat.pol_ind = m.pol.ind; R.mat.pol_w = m.pol.w; R.mat.pol_inv_h = m.pol.inv_h;
R.mat.d_ind = m.d.ind; R.mat.d_w = m.d.w; R.mat.d_inv_h = m.d.inv_h;
R.mat.d_x_bar = m.d.x_bar; R.mat.d_hist = m.d.hist;
R.mat.Phi_e_nnz = nnz(m.pol.Phi_e); R.mat.Phi_e_sum = sum(m.pol.Phi_e, 'all');
[i, j, v] = find(m.pol.Phi_e); k = round(linspace(1, numel(v), min(200, numel(v))));
R.mat.Phi_e_samples = [i(k) j(k) v(k)];
R.mat.Phi_nnz = nnz(m.d.Phi); R.mat.Phi_sum = sum(m.d.Phi, 'all');
R.mat.G = m.G; R.mat.dG = full(m.dG); R.mat.y = m.y;
R.mat.F = m.F; R.mat.dF_size = size(m.dF);
R.mat.dF_samples = m.dF(:, :, round(linspace(1, size(m.dF, 3), 5)));
R.mat.dF_sample_idx = round(linspace(1, size(m.dF, 3), 5));
if exist('Ix_dummy', 'var'), end
R.mat.Ix = m.pol.x_bar_dash * m.d.Phi * m.d.hist;
R.params_after = M_.params;

% ---- rouwenhorst
[yr, pr, Pr] = rouwenhorst(0.966, 0.5, 7, 1e-12, 1e5);
R.rouwenhorst = struct('y', yr, 'p', pr, 'P', Pr);
[yr, pr, Pr] = rouwenhorst(0.9, 0.3, 4, 1e-10, 1000);
R.rouwenhorst2 = struct('y', yr, 'p', pr, 'P', Pr);

R = unsparse(R);
fid = fopen(sprintf('ref_%s.json', mdl), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);

function x = unsparse(x)
    if isstruct(x)
        for k = 1:numel(x)
            f = fieldnames(x);
            for i = 1:numel(f)
                x(k).(f{i}) = unsparse(x(k).(f{i}));
            end
        end
    elseif issparse(x)
        x = full(x);
    end
end
