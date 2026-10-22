% Reference values for the optimal-policy port (stream P): runs every model of this
% folder with Dynare 7.2 and writes ref_<model>.json.
% Run from a private copy of reference/policy (Dynare writes +<model>/ and <model>/ here):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_policy.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
models = {'gali_discretion', 'gali_ramsey', 'gali_osr', 'ramsey_cmr', 'ramsey_cmr2', 'gali_osr_simplex', ...
          'ramsey_cmr_ssm', 'ramsey_cmr_o2', 'cmr_osr_o2', 'gali_discretion2'};
if exist('only_models', 'var')
    models = only_models;
end
for k = 1:numel(models)
    name = models{k};
    if ~isfile([name '.mod'])
        continue
    end
    clearvars -global
    global M_ oo_ options_
    osr_init = struct('phi_pi', 1.5, 'phi_y', 0.1); % initial osr_params of cmr_osr_o2
    eval(['dynare ' name ' noclearall nolog']);
    r = struct();
    r.endo_names = M_.endo_names;
    r.param_names = M_.param_names;
    r.params = M_.params;
    r.lead_lag_incidence = M_.lead_lag_incidence;
    r.eq_nbr = M_.eq_nbr;
    r.orig_endo_nbr = M_.orig_endo_nbr;
    r.Sigma_e = M_.Sigma_e;
    r.steady_state = oo_.steady_state;
    dr = oo_.dr;
    r.ys = dr.ys;
    r.ghx = dr.ghx;
    r.ghu = dr.ghu;
    % order 2 
    for f = {'ghxx', 'ghxu', 'ghuu', 'ghs2'}
        if isfield(dr, f{1})
            r.(f{1}) = dr.(f{1});
        end
    end
    r.order = options_.order;
    r.order_var = dr.order_var;
    if isfield(dr, 'state_var')
        r.state_var = dr.state_var;
    end
    r.nstatic = M_.nstatic;
    r.nspred = M_.nspred;
    if isfield(oo_, 'var')
        r.var = oo_.var;
    end
    if isfield(oo_, 'mean')
        r.mean = oo_.mean;
    end
    r.var_list = oo_.var_list;
    if isfield(oo_, 'planner_objective_value')
        r.planner_objective_value = oo_.planner_objective_value;
    end
    if isfield(oo_, 'osr')
        r.osr_objective = oo_.osr.objective_function;
        r.osr_params = oo_.osr.optim_params;
        r.osr_param_names = M_.osr.param_names;
        r.osr_info = oo_.osr.optimization_info;
    end
    if isfield(oo_, 'osr') && options_.order == 2
        % second-order welfare at the optimum (oo_.dr is the optimal rule) and at the calibration
        opts = options_; opts.noprint = 1;
        r.osr_welfare_opt = evaluate_planner_objective(M_, opts, oo_);
        M0 = M_;
        p0 = struct2cell(oo_.osr.optim_params);
        names = fieldnames(oo_.osr.optim_params);
        r.osr_init_params = zeros(numel(names), 1);
        for q = 1:numel(names)
            iq = strmatch(names{q}, M_.param_names, 'exact');
            r.osr_init_params(q) = osr_init.(names{q});
            M0.params(iq) = osr_init.(names{q});
        end
        oo0 = oo_;
        [oo0.dr, info0, M0.params] = resol(0, M0, opts, oo0.dr, oo0.steady_state, oo0.exo_steady_state, oo0.exo_det_steady_state);
        r.osr_welfare_init = evaluate_planner_objective(M0, opts, oo0);
        r.osr_init_ghxx = oo0.dr.ghxx;
        r.osr_init_ghs2 = oo0.dr.ghs2;
    end
    % static and dynamic residual/Jacobian at the steady state (checks the FOCs)
    ys = dr.ys;
    x = zeros(1, M_.exo_nbr);
    r.static_resid = feval([M_.fname '.static_resid'], ys, x, M_.params);
    r.static_g1 = full(feval([M_.fname '.static_g1'], ys, x, M_.params, M_.static_g1_sparse_rowval, M_.static_g1_sparse_colval, M_.static_g1_sparse_colptr));
    y3 = [ys; ys; ys];
    r.dynamic_g1 = full(feval([M_.fname '.dynamic_g1'], y3, x, M_.params, ys, M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr));
    fid = fopen(['ref_' name '.json'], 'w');
    fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', true));
    fclose(fid);
end
