% dump_het.m -- Dynare 7.2 reference values of the examples/heterogeneity and
% examples/optimal_policy examples (tests/examples72_het.rs).
% Run in a PRIVATE copy holding the example files (and this script), one MATLAB session per model:
%   matlab -batch "mdl='krusell_smith_1998'; run('dump_het.m')"   -> ref_<mdl>.json
% (MATLAB may segfault on exit after the dump; the JSON is complete.)
% Steady-state examples also save <mdl>_initial_guess.mat (the verbatim-built initial guess:
% initial_guess_ref.mat in reference/examples72/<mdl>/).
% nk_ramsey_osr's OSR variant: a copy of the .mod with '@#define Ramsey=0' as its first line,
% dumped as nk_osr; reference/examples72/nk_ramsey_*/ref.json keep a subset of the fields
% (osr_variant = the nk_osr dump; exo_simul_first = oo_.exo_simul(1,:)).
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath')); cd(here);
if ~exist('mdl','var'), error('mdl'); end
if ~exist('dopts','var'), dopts = ''; end
t0 = tic;
eval(sprintf('dynare %s nointeractive nograph %s', mdl, dopts));
global M_ options_ oo_
R = struct();
R.time = toc(t0);
if exist('initial_guess', 'var')
  save(sprintf('%s_initial_guess.mat', mdl), 'initial_guess', '-v7');
end
R.params = M_.params; R.param_names = M_.param_names;
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names;
if isfield(oo_,'steady_state'), R.steady_state = oo_.steady_state; end
if isfield(oo_,'dr')
  for f = {'ghx','ghu','ghxx','ghxu','ghuu','ghs2','order_var','ys'}
    if isfield(oo_.dr, f{1}), R.dr.(f{1}) = oo_.dr.(f{1}); end
  end
end
for f = {'var','mean','gamma_y','irfs','osr','planner_objective_value','endo_simul','exo_simul'}
  if isfield(oo_, f{1}), R.(f{1}) = oo_.(f{1}); end
end
if isfield(oo_,'heterogeneity')
  h = oo_.heterogeneity;
  if isfield(h,'steady_state')
    ss = h.steady_state;
    R.het.agg = ss.agg;
    fn = fieldnames(ss.pol.values);
    for i=1:numel(fn), v = ss.pol.values.(fn{i}); R.het.pol.(fn{i}) = struct('size',size(v),'sum',sum(v(:)),'first',v(1:min(10,end)),'last',v(end)); end
    if isfield(ss,'d'), d = ss.d.hist; R.het.d = struct('size',size(d),'sum',sum(d(:)),'max',max(d(:)), 'first', d(1:min(10,end))); end
    steady_state = ss; save(sprintf('%s_ss.mat', mdl), 'steady_state', '-v7');
  end
  if isfield(h,'dr') && isfield(h.dr,'G')
    T = options_.heterogeneity.solve.truncation_horizon; R.het.T = T;
    G = h.dr.G; fn = fieldnames(G);
    for i=1:numel(fn), fs = fieldnames(G.(fn{i})); for j=1:numel(fs), g = G.(fn{i}).(fs{j}); R.het.G.(fn{i}).(fs{j}) = g(1:min(5,end),1)'; end, end
  end
  if isfield(oo_,'endo_simul') && ~isempty(oo_.endo_simul) && options_.periods > 0 && isempty(M_.det_shocks)
    NP = min(40, options_.periods);
    set_dynare_seed_local_options([],false,'default');
    total = options_.periods + T - 1;
    i_exo_var = setdiff(1:M_.exo_nbr, find(diag(M_.Sigma_e) == 0));
    nxs = length(i_exo_var);
    eps_draws = zeros(M_.exo_nbr, total);
    chol_S = chol(M_.Sigma_e(i_exo_var, i_exo_var));
    eps_draws(i_exo_var, :) = (randn(nxs, total)' * chol_S)';
    R.het.NP = NP;
    R.het.eps_draws = eps_draws(:, 1:NP+T-1);
    R.endo_simul = oo_.endo_simul(:, 1:NP);
    R.het.Sigma_e = M_.Sigma_e;
  end
end
R = unsparse_local(R);
fid = fopen(sprintf('ref_%s.json', mdl), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('DUMPED %s in %.1fs\n', mdl, R.time);
function x = unsparse_local(x)
  if issparse(x), x = full(x);
  elseif isstruct(x)
    for k = 1:numel(x), fn = fieldnames(x(k)); for i=1:numel(fn), x(k).(fn{i}) = unsparse_local(x(k).(fn{i})); end, end
  elseif iscell(x), for i=1:numel(x), x{i} = unsparse_local(x{i}); end
  end
end
