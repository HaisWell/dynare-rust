% dump_mom.m -- method-of-moments reference values of Dynare (7.2 by default, see DYNARE_VERSION) (stream MoM).
%
% 1. Simulates the data once (rbc_mom.mod -DGEN=1/2, order 1/2) into rbc_mom_data1/2.csv.
% 2. Runs rbc_mom.mod for each case (-DCASE=k) and writes ref_rbc_case<k>.json with oo_.mom,
%    the objective / model moments at fixed points (identity and final weighting matrix),
%    the SMM shock series (options_mom_.mom.shock_series) and the console log.
%
% Run from a PRIVATE copy of reference/mom (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_mom.m')"
% Optional: cases = [1 4]; before run(...) to redo some cases only.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('cases', 'var'), cases = 1:6; end

for gen = 1:2
    fn = sprintf('rbc_mom_data%d.csv', gen);
    if ~isfile(fn)
        eval(sprintf('dynare rbc_mom -DGEN=%d nointeractive', gen));
        names = {'yo', 'co', 'lo', 'io'};
        idx = cellfun(@(n) find(strcmp(M_.endo_names, n)), names);
        Y = oo_.endo_simul(idx, :)';
        fid = fopen(fn, 'w');
        fprintf(fid, '%s,%s,%s,%s\n', names{:});
        fprintf(fid, '%.17g,%.17g,%.17g,%.17g\n', Y');
        fclose(fid);
    end
end

% Dynare 7.1 bug (still present in 7.2): GMM at order 3 calls k_order_pert with options_mom_, which has no
% `verbosity` field. A patched copy in this (private) folder shadows the installed one.
src = fileread(fullfile(fileparts(which('dynare')), 'stochastic_solver', 'k_order_pert.m'));
src = regexprep(src, '\ninfo = 0;', '\ninfo = 0;\nif ~isfield(options_, ''verbosity''), options_.verbosity = 0; end', 'once');
fid = fopen('k_order_pert.m', 'w'); fprintf(fid, '%s', src); fclose(fid);
rehash;

for k = cases
    clear global M_ options_ oo_ estim_params_ bayestopt_ options_mom_
    eval(sprintf('dynare rbc_mom -DCASE=%d nointeractive', k));
    S = load(fullfile('rbc_mom', 'method_of_moments', 'rbc_mom_mom_results.mat'));
    om = S.options_mom_;
    R = struct();
    R.case = k;
    R.mom_method = om.mom.mom_method;
    R.order = om.order; R.pruning = om.pruning; R.prefilter = om.prefilter;
    R.nobs = om.nobs;
    R.param_names = S.bayestopt_.name;
    R.matched_moments = S.M_.matched_moments;
    R.obs_var = om.mom.obs_var;
    R.order_var = S.oo_.dr.order_var;
    R.mom = S.oo_.mom;
    if isfield(R.mom, 'verbose') && isempty(R.mom.verbose), R.mom = rmfield(R.mom, 'verbose'); end
    if strcmp(om.mom.mom_method, 'SMM')
        R.shock_series = om.mom.shock_series;
        R.long = om.mom.long;
        R.burnin = om.mom.burnin;
    end
    % objective and model moments at fixed points
    xm = S.oo_.mom.([lower(om.mom.mom_method) '_mode']);
    np = numel(S.bayestopt_.name);
    BoundsInfo.lb = -Inf(np, 1); BoundsInfo.ub = Inf(np, 1);
    x0 = S.bayestopt_.p1; % not used; points below
    pts = {};
    xinit = get_x0(S);
    pts{end+1} = xinit;
    pts{end+1} = xinit .* (1 + 0.03*(1:np)');
    xfin = get_xparam(S);
    pts{end+1} = xfin;
    R.points = {};
    wi_id.Sw = eye(om.mom.mom_nbr); wi_id.Woptflag = false;
    wi_fin = S.oo_.mom.weighting_info;
    om.mom.vector_output = false; om.mom.compute_derivs = false;
    for j = 1:numel(pts)
        x = pts{j};
        P = struct('x', x);
        [P.fval_identity, info, ~, ~, ~, P.Q_identity, P.model_moments] = mom.objective_function(x, S.oo_.mom.data_moments, wi_id, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
        P.info = info(1);
        P.fval_final = mom.objective_function(x, S.oo_.mom.data_moments, wi_fin, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
        R.points{end+1} = P;
    end
    % optimal weighting matrix from the data moments (stage-1 OPTIMAL / DIAGONAL)
    R.W_opt_data = mom.optimal_weighting_matrix(S.oo_.mom.m_data, S.oo_.mom.data_moments, om.mom.bartlett_kernel_lag);
    % standard errors at the final point with the final weighting matrix
    [R.se_at_final, R.asympt_cov_at_final] = mom.standard_errors(xfin, @mom.objective_function, S.oo_.mom.model_moments, [], S.oo_.mom.m_data, S.oo_.mom.data_moments, wi_fin, om, S.M_, S.estim_params_, S.bayestopt_, BoundsInfo, S.oo_.dr, S.oo_.steady_state, S.oo_.exo_steady_state, S.oo_.exo_det_steady_state);
    copyfile('rbc_mom.log', sprintf('rbc_case%d_matlab.log', k));
    fid = fopen(sprintf('ref_rbc_case%d.json', k), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
    fclose(fid);
end

function x = get_xparam(S)
    % final estimate in xparam1 order (stderr first, then corr, then parameters)
    ep = S.estim_params_; M = S.M_; x = [];
    for i = 1:ep.nvx, x(end+1, 1) = sqrt(M.Sigma_e(ep.var_exo(i, 1), ep.var_exo(i, 1))); end
    for i = 1:ep.np, x(end+1, 1) = M.params(ep.param_vals(i, 1)); end
end

function x = get_x0(S)
    ep = S.estim_params_; x = [];
    for i = 1:ep.nvx, x(end+1, 1) = ep.var_exo(i, 2); end
    for i = 1:ep.np, x(end+1, 1) = ep.param_vals(i, 2); end
end
