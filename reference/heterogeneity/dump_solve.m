% dump_solve.m -- reference values of Dynare (DYNARE_VERSION, default 7.2) for heterogeneity_solve / heterogeneity_simulate.
% Run from a PRIVATE copy of reference/heterogeneity, one fresh MATLAB session per model:
%   matlab -batch "mdl='ks'; run('dump_solve.m')"      (also 'hank1', and 'ksnews': ks with news shocks)
% writes ref_solve_<mdl>.json (and <mdl>_ss.mat: oo_.heterogeneity.steady_state, the exact input of the
% Rust test, and <mdl>_s.mod: the model plus the two commands).
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
dynare_config;
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('mdl', 'var'), mdl = 'ks'; end
global M_ options_ oo_ DBG
T = 40; IRF = 30; PER = 25;
src = mdl;
news = strcmp(mdl, 'ksnews');
if news, src = 'ks'; end
txt = fileread([src '.mod']);
if news
    txt = strrep(txt, 'var Z; stderr 0.01;', 'var Z; stderr 0.01; var Z; periods 2:6; values 0.01;');
end
txt = [txt sprintf('\nheterogeneity_solve(truncation_horizon = %d);\n', T)];
if news
    txt = [txt sprintf('heterogeneity_simulate;\n')];
else
    txt = [txt sprintf('heterogeneity_simulate(irf = %d, periods = %d);\n', IRF, PER)];
end
fid = fopen([mdl '_s.mod'], 'w'); fprintf(fid, '%s', txt); fclose(fid);
eval(sprintf('dynare %s_s nointeractive nograph', mdl));

global DBG
flat = @(x) reshape(x, 1, []);
nd = @(x) struct('size', size(x), 'data', flat(x));
steady_state = oo_.heterogeneity.steady_state; %#ok<NASGU>
save([mdl '_ss.mat'], 'steady_state', '-v7');
R = struct();
R.T = options_.heterogeneity.solve.truncation_horizon; R.IRF = options_.irf; R.PER = options_.periods;
R.params_after = M_.params;
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names;
dr = oo_.heterogeneity.dr;
R.dr = dr;
fn = fieldnames(dr.curlyDs);
for i = 1:numel(fn), R.curlyDs.(fn{i}) = nd(dr.curlyDs.(fn{i})); end
R.dr = rmfield(R.dr, 'curlyDs');
% intermediates of solve.m
s = fileread([droot '/matlab/+heterogeneity/solve.m']);
s = strrep(s, 'function dr = solve(', 'function dr = solve_dbg(');
s = strrep(s, 'expectations = compute_expected_pol(mat, sizes, indices, T, N_Ix);', ...
    'expectations = compute_expected_pol(mat, sizes, indices, T, N_Ix); global DBG; DBG.ir = impulse_responses; DBG.expectations = expectations;');
fid = fopen('solve_dbg.m', 'w'); fprintf(fid, '%s', s); fclose(fid);
rehash;
dr2 = solve_dbg(M_, options_.heterogeneity.solve, oo_.heterogeneity);
R.dbg_check = max(abs(dr2.G.(M_.endo_names{1}).(M_.exo_names{1}) - dr.G.(M_.endo_names{1}).(M_.exo_names{1})), [], 'all');
ir = DBG.ir;
R.ir_size = size(ir);
R.ir_s1 = nd(ir(:,:,:,1)); R.ir_s2 = nd(ir(:,:,:,2)); R.ir_s7 = nd(ir(:,:,:,7));
R.exp_size = size(DBG.expectations);
R.exp_s1 = nd(DBG.expectations(:,:,1)); R.exp_s2 = nd(DBG.expectations(:,:,2)); R.exp_s9 = nd(DBG.expectations(:,:,9));
R.sizes = oo_.heterogeneity.sizes;
if isfield(oo_, 'irfs'), R.irfs = oo_.irfs; end
R.endo_simul = oo_.endo_simul;
R.y_ss = oo_.heterogeneity.mat.y;
if ~news
    set_dynare_seed_local_options([],false,'default');
    total = options_.periods + options_.heterogeneity.solve.truncation_horizon - 1;
    i_exo_var = setdiff(1:M_.exo_nbr, find(diag(M_.Sigma_e) == 0));
    nxs = length(i_exo_var);
    eps_draws = zeros(M_.exo_nbr, total);
    chol_S = chol(M_.Sigma_e(i_exo_var, i_exo_var));
    eps_draws(i_exo_var, :) = (randn(nxs, total)' * chol_S)';
    R.eps_draws = eps_draws;
    R.Sigma_e = M_.Sigma_e;
else
    R.exo_simul = oo_.exo_simul;
end
R = unsparse(R);
fid = fopen(sprintf('ref_solve_%s.json', mdl), 'w');
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
