% dump_analytic.m -- analytic Jacobians / standard errors of Dynare's (7.2 by default, see DYNARE_VERSION) method_of_moments (GMM).
%
% Runs rbc_mom.mod -DANALYTIC=1 -DCASE=k (k = 1 GMM order 1, 2 order 2, 3 order 3 with
% analytic_jacobian, 6 penalized order 1 with analytic_jacobian) and writes
% ref_rbc_analytic_case<k>.json: the run's standard errors, and at three points
% (initial, perturbed, final) the objective with compute_derivs = true: model moments, their
% analytic Jacobian (model_moments_params_derivs), the gradient df (scalar objective) and the
% residual Jacobian (vector_output). rbc_mom_data1/2.csv must exist (dump_mom.m).
% Run from a PRIVATE copy of reference/mom:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_analytic.m')" > log.txt 2>&1
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('cases', 'var'), cases = [1 2 3 6]; end

src = fileread(fullfile(fileparts(which('dynare')), 'stochastic_solver', 'k_order_pert.m'));
src = regexprep(src, '\ninfo = 0;', '\ninfo = 0;\nif ~isfield(options_, ''verbosity''), options_.verbosity = 0; end', 'once');
fid = fopen('k_order_pert.m', 'w'); fprintf(fid, '%s', src); fclose(fid);
rehash;

for k = cases
    clear global M_ options_ oo_ estim_params_ bayestopt_ options_mom_
    eval(sprintf('dynare rbc_mom -DCASE=%d -DANALYTIC=1 nointeractive', k));
    S = load(fullfile('rbc_mom', 'method_of_moments', 'rbc_mom_mom_results.mat'));
    om = S.options_mom_;
    R = struct();
    R.case = k;
    R.analytic_standard_errors = om.mom.analytic_standard_errors;
    R.analytic_jacobian = om.mom.analytic_jacobian;
    R.mom = S.oo_.mom;
    if isfield(R.mom, 'verbose') && isempty(R.mom.verbose), R.mom = rmfield(R.mom, 'verbose'); end
    np = numel(S.bayestopt_.name);
    BoundsInfo.lb = -Inf(np, 1); BoundsInfo.ub = Inf(np, 1);
    xinit = get_x0(S);
    xfin = get_xparam(S);
    pts = {xinit, xinit .* (1 + 0.03*(1:np)'), xfin};
    wi_fin = S.oo_.mom.weighting_info;
    om.mom.compute_derivs = true; om.mom.analytic_standard_errors = true; om.mom.analytic_jacobian = true;
    R.points = {};
    for j = 1:numel(pts)
        x = pts{j};
        P = struct('x', x);
        om.mom.vector_output = false;
        [P.fval, info, ~, P.df, ~, P.Q, P.model_moments, P.D] = mom.objective_function(x, S.oo_.mom.data_moments, wi_fin, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
        P.info = info(1);
        om.mom.vector_output = true;
        [~, ~, ~, P.df_vec] = mom.objective_function(x, S.oo_.mom.data_moments, wi_fin, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
        R.points{end+1} = P;
    end
    om.mom.vector_output = false;
    [R.se_at_final, R.asympt_cov_at_final] = mom.standard_errors(xfin, @mom.objective_function, R.points{3}.model_moments, R.points{3}.D, S.oo_.mom.m_data, S.oo_.mom.data_moments, wi_fin, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
    copyfile('rbc_mom.log', sprintf('rbc_analytic_case%d_matlab.log', k));
    fid = fopen(sprintf('ref_rbc_analytic_case%d.json', k), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
    fclose(fid);
end

function x = get_xparam(S)
    ep = S.estim_params_; M = S.M_; x = [];
    for i = 1:ep.nvx, x(end+1, 1) = sqrt(M.Sigma_e(ep.var_exo(i, 1), ep.var_exo(i, 1))); end
    for i = 1:ep.np, x(end+1, 1) = M.params(ep.param_vals(i, 1)); end
end

function x = get_x0(S)
    ep = S.estim_params_; x = [];
    for i = 1:ep.nvx, x(end+1, 1) = ep.var_exo(i, 2); end
    for i = 1:ep.np, x(end+1, 1) = ep.param_vals(i, 2); end
end
