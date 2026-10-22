% dump_nonlinear.m -- reference values of Dynare 7.2's nonlinear filters (stream PF):
% non_linear_dsge_likelihood.m with every filter of nonlinear-filters/ on rbcnl.mod
% (RBC at order 2 and 3, measurement errors on y and c, simulated data), with the
% uniform/normal draws of every stochastic filter logged (shadow/rand.m, shadow/randn.m) so
% that dynare-rust can replay them; plus local_state_space_iteration_3 test vectors and
% per-evaluation timings.
%
% Run from a PRIVATE copy of reference/nonlinear (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_nonlinear.m')"
% writes rbcnl_data.csv, ref_rbcnl_o2.json, ref_rbcnl_o3.json, ref_timing.json.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
global PF_LOG
PF_LOG = struct('on', false, 'normals', {{}}, 'uniforms', {{}});
addpath(fullfile(here, 'shadow'), '-begin');
warning('off', 'MATLAB:dispatcher:nameConflict');

% ---- simulated data ---------------------------------------------------------------
if ~isfile('rbcnl_data.csv')
    dynare rbcnl -DSIMULATE nointeractive
    global M_ oo_
    T = 60;
    sim = oo_.endo_simul(:, end-T+1:end);
    rng(7);
    yobs = sim(strcmp(M_.endo_names, 'y'), :)' + 0.004*randn(T, 1);
    cobs = sim(strcmp(M_.endo_names, 'c'), :)' + 0.003*randn(T, 1);
    fid = fopen('rbcnl_data.csv', 'w');
    fprintf(fid, 'y,c\n');
    fprintf(fid, '%.17g,%.17g\n', [yobs cobs]');
    fclose(fid);
    clear global
    global PF_LOG
    PF_LOG = struct('on', false, 'normals', {{}}, 'uniforms', {{}});
end

timing = struct();
for ORDER = [2 3]
    % (Dynare 7.1 then fails in display_estimation_results_table with mode_compute=0 and no
    % Hessian; everything needed below is set up by then.)
    try
        eval(sprintf('dynare rbcnl -DORDER=%d nointeractive', ORDER));
    catch e
        disp(e.message);
    end
    global M_ options_ oo_ bayestopt_ estim_params_ PF_LOG
    PF_LOG = struct('on', false, 'normals', {{}}, 'uniforms', {{}});
    options_.particle = check_particle_filter_options(options_.particle);
    [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, BoundsInfo] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    base = options_.particle;
    x0 = xparam1;
    x1 = x0 .* [0.98; 1.1; 1.2; 0.9; 1.15];
    R = struct();
    R.order = ORDER;
    R.x0 = x0; R.x1 = x1;
    R.data = dataset_.data;
    R.param_names = bayestopt_.name;
    % ReducedForm at x0 (as non_linear_dsge_likelihood builds it)
    rf = reduced_form(x0, M_, options_, oo_, bayestopt_, estim_params_);
    R.rf = struct('steadystate', rf.steadystate, 'constant', rf.constant, ...
        'state_variables_steady_state', rf.state_variables_steady_state, 'Q', rf.Q, 'H', rf.H, ...
        'mf0', rf.mf0, 'mf1', rf.mf1, 'ghx', rf.ghx, 'ghu', rf.ghu, 'ghxx', rf.ghxx, 'ghuu', rf.ghuu, ...
        'ghxu', rf.ghxu, 'ghs2', rf.ghs2, 'StateVectorMean', rf.StateVectorMean, ...
        'StateVectorVariance', rf.StateVectorVariance, 'restrict_var_list', oo_.dr.restrict_var_list);
    if ORDER == 3
        % local_state_space_iteration_3 test vectors (with and without pruning)
        rng(11);
        ns = numel(rf.mf0); nu = size(rf.ghu, 2); m = size(rf.ghx, 1);
        yhat = 0.05*randn(ns, 4); yhat3 = 0.05*randn(3*ns, 4); ep = 0.02*randn(nu, 4);
        y = local_state_space_iteration_3(yhat, ep, rf.ghx, rf.ghu, rf.ghxx, rf.ghuu, rf.ghxu, rf.ghs2, ...
            rf.ghxxx, rf.ghuuu, rf.ghxxu, rf.ghxuu, rf.ghxss, rf.ghuss, rf.steadystate, 1, false);
        [yp, yp_] = local_state_space_iteration_3(yhat3, ep, rf.ghx, rf.ghu, rf.ghxx, rf.ghuu, rf.ghxu, rf.ghs2, ...
            rf.ghxxx, rf.ghuuu, rf.ghxxu, rf.ghxuu, rf.ghxss, rf.ghuss, rf.steadystate, 1, true);
        R.lssi3 = struct('yhat', yhat, 'yhat3', yhat3, 'epsilon', ep, 'y', y, 'y_pruned', yp, 'y_pruned_state', yp_, ...
            'ghxxx', rf.ghxxx, 'ghuuu', rf.ghuuu, 'ghxxu', rf.ghxxu, 'ghxuu', rf.ghxuu, 'ghxss', rf.ghxss, 'ghuss', rf.ghuss);
    end

    % ---- the cases ------------------------------------------------------------------
    C = {};
    C{end+1} = {'sis_kitagawa', {'filter_algorithm','sis'}};
    C{end+1} = {'sis_stratified_generic', {'filter_algorithm','sis','resampling.method','stratified','resampling.status','generic','resampling.threshold',0.5}};
    C{end+1} = {'sis_none', {'filter_algorithm','sis','resampling.status','none'}};
    C{end+1} = {'sis_smooth', {'filter_algorithm','sis','resampling.method','smooth'}};
    C{end+1} = {'sis_pruning', {'filter_algorithm','sis','pruning',true}};
    C{end+1} = {'apf_kitagawa', {'filter_algorithm','apf'}};
    C{end+1} = {'apf_stratified', {'filter_algorithm','apf','resampling.method','stratified'}};
    C{end+1} = {'gf_cubature', {'filter_algorithm','gf','proposal','cubature','distribution','cubature'}};
    C{end+1} = {'gf_unscented', {'filter_algorithm','gf','proposal','unscented','distribution','unscented'}};
    C{end+1} = {'gf_montecarlo', {'filter_algorithm','gf','proposal','montecarlo','distribution','montecarlo','number_of_particles',100}};
    C{end+1} = {'gmf_cubature', {'filter_algorithm','gmf','proposal','cubature','distribution','cubature'}};
    C{end+1} = {'gmf_unscented', {'filter_algorithm','gmf','proposal','unscented','distribution','unscented','mixture_state_variables',3,'mixture_structural_shocks',2}};
    C{end+1} = {'cpf_cubature', {'filter_algorithm','cpf','proposal','cubature','number_of_particles',100}};
    C{end+1} = {'cpf_unscented', {'filter_algorithm','cpf','proposal','unscented','number_of_particles',100}};
    C{end+1} = {'cpf_unscented_mjp', {'filter_algorithm','cpf','proposal','unscented','cpf_weights','murrayjonesparslow','number_of_particles',100}};
    C{end+1} = {'cpf_montecarlo', {'filter_algorithm','cpf','proposal','montecarlo','number_of_particles',50}};
    C{end+1} = {'nlkf_cubature', {'filter_algorithm','nlkf','proposal','cubature'}};
    C{end+1} = {'nlkf_unscented', {'filter_algorithm','nlkf','proposal','unscented'}};
    C{end+1} = {'nlkf_montecarlo', {'filter_algorithm','nlkf','proposal','montecarlo','number_of_particles',100}};
    if ORDER == 3
        C = C([1 3 5 6 8 9 13 14 17 18]);
    end
    R.cases = {};
    Y = transpose(dataset_.data);
    start = options_.presample + 1;
    for k = 1:numel(C)
        name = C{k}{1}; kv = C{k}{2};
        o = options_; o.particle = apply_case(base, kv);
        o.particle = check_particle_filter_options(o.particle);
        S = struct('name', name, 'options', {kv});
        for ip = 1:2
            x = x0; if ip == 2, x = x1; end
            rfx = reduced_form(x, M_, o, oo_, bayestopt_, estim_params_);
            PF_LOG.on = true; PF_LOG.normals = {}; PF_LOG.uniforms = {};
            try
                [LIK, lik] = feval(o.particle.algorithm, rfx, Y, start, o.particle, o.threads, o, M_);
                PF_LOG.on = false;
                [fval, info] = non_linear_dsge_likelihood(x, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, ...
                    BoundsInfo, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
                P = struct('LIK', LIK, 'lik', lik, 'fval', fval, 'info', info(1), 'error', '');
            catch err
                % a MATLAB error inside the filter (e.g. a rank-deficient square root)
                PF_LOG.on = false;
                P = struct('LIK', NaN, 'lik', [], 'fval', NaN, 'info', -1, 'error', err.message);
            end
            if ip == 1
                P.normals = vertcat(PF_LOG.normals{:});
                P.uniforms = vertcat(PF_LOG.uniforms{:});
            end
            S.(sprintf('p%d', ip-1)) = P;
        end
        R.cases{end+1} = S;
        fprintf('dump_nonlinear: order %d %-24s fval(x0) = %.10f  fval(x1) = %.10f\n', ORDER, name, S.p0.fval, S.p1.fval);
    end
    fid = fopen(sprintf('ref_rbcnl_o%d.json', ORDER), 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
    fclose(fid);

    % ---- timings: one evaluation of non_linear_dsge_likelihood ------------------------
    T_ = {{'sis_5000', {'filter_algorithm','sis','number_of_particles',5000}}, ...
          {'sis_20000', {'filter_algorithm','sis','number_of_particles',20000}}, ...
          {'apf_5000', {'filter_algorithm','apf','number_of_particles',5000}}, ...
          {'gf_cubature', {'filter_algorithm','gf','proposal','cubature','distribution','cubature'}}, ...
          {'nlkf_unscented', {'filter_algorithm','nlkf','proposal','unscented'}}, ...
          {'cpf_unscented_500', {'filter_algorithm','cpf','proposal','unscented','number_of_particles',500}}, ...
          {'gmf_cubature', {'filter_algorithm','gmf','proposal','cubature','distribution','cubature'}}};
    for k = 1:numel(T_)
        o = options_; o.particle = check_particle_filter_options(apply_case(base, T_{k}{2}));
        f = @() non_linear_dsge_likelihood(x0, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, ...
                BoundsInfo, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
        f(); n = 0; t0 = tic;
        while toc(t0) < 3 || n < 2
            f(); n = n + 1;
        end
        timing.(sprintf('o%d_%s', ORDER, T_{k}{1})) = toc(t0)/n;
        fprintf('dump_nonlinear: timing order %d %-20s %.4f s per evaluation\n', ORDER, T_{k}{1}, toc(t0)/n);
    end
end
fid = fopen('ref_timing.json', 'w');
fprintf(fid, '%s', jsonencode(timing));
fclose(fid);
disp('dump_nonlinear: done');

function p = apply_case(p, kv)
for i = 1:2:numel(kv)
    key = kv{i}; v = kv{i+1};
    switch key
        case 'resampling.method'
            p.resampling.method.kitagawa = false; p.resampling.method.stratified = false; p.resampling.method.smooth = false;
            p.resampling.method.(v) = true;
        case 'resampling.status'
            p.resampling.status.systematic = false; p.resampling.status.none = false; p.resampling.status.generic = false;
            p.resampling.status.(v) = true;
        case 'resampling.threshold'
            p.resampling.threshold = v;
        case 'proposal'
            p.proposal_approximation.cubature = false; p.proposal_approximation.unscented = false; p.proposal_approximation.montecarlo = false;
            p.proposal_approximation.(v) = true;
        case 'distribution'
            p.distribution_approximation.cubature = false; p.distribution_approximation.unscented = false; p.distribution_approximation.montecarlo = false;
            p.distribution_approximation.(v) = true;
        case 'cpf_weights'
            p.cpf_weights_method.amisanotristani = false; p.cpf_weights_method.murrayjonesparslow = false;
            p.cpf_weights_method.(v) = true;
        otherwise
            p.(key) = v;
    end
end
end

function ReducedForm = reduced_form(xparam1, M_, options_, oo_, bayestopt_, estim_params_)
% The ReducedForm of non_linear_dsge_likelihood.m (initialization = 1).
M_ = set_all_parameters(xparam1, estim_params_, M_);
[dr, info, M_.params] = resol(0, M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
if info(1), error('resol failed'); end
mf0 = bayestopt_.mf0;
restrict_variables_idx = dr.restrict_var_list;
state_variables_idx = restrict_variables_idx(mf0);
ReducedForm.steadystate = dr.ys(dr.order_var(restrict_variables_idx));
ReducedForm.constant = ReducedForm.steadystate + .5*dr.ghs2(restrict_variables_idx);
ReducedForm.state_variables_steady_state = dr.ys(dr.order_var(state_variables_idx));
ReducedForm.Q = M_.Sigma_e;
ReducedForm.H = M_.H;
ReducedForm.mf0 = mf0;
ReducedForm.mf1 = bayestopt_.mf1;
ReducedForm.use_k_order_solver = false;
ReducedForm.ghx  = dr.ghx(restrict_variables_idx,:);
ReducedForm.ghu  = dr.ghu(restrict_variables_idx,:);
ReducedForm.ghxx = dr.ghxx(restrict_variables_idx,:);
ReducedForm.ghuu = dr.ghuu(restrict_variables_idx,:);
ReducedForm.ghxu = dr.ghxu(restrict_variables_idx,:);
ReducedForm.ghs2 = dr.ghs2(restrict_variables_idx,:);
if options_.order==3
    ReducedForm.ghxxx = dr.ghxxx(restrict_variables_idx,:);
    ReducedForm.ghuuu = dr.ghuuu(restrict_variables_idx,:);
    ReducedForm.ghxxu = dr.ghxxu(restrict_variables_idx,:);
    ReducedForm.ghxuu = dr.ghxuu(restrict_variables_idx,:);
    ReducedForm.ghxss = dr.ghxss(restrict_variables_idx,:);
    ReducedForm.ghuss = dr.ghuss(restrict_variables_idx,:);
end
ReducedForm.StateVectorMean = ReducedForm.constant(mf0);
[A,B] = kalman_transition_matrix(dr,dr.restrict_var_list,dr.restrict_columns);
V = lyapunov_symm(A, B*ReducedForm.Q*B', options_.lyapunov_fixed_point_tol, ...
                  options_.qz_criterium, options_.lyapunov_complex_threshold, [], options_.debug);
ReducedForm.StateVectorVariance = V(mf0,mf0);
end
