% dump_is.m -- Dynare 7.2 reference values for estimate_initial_states_endogenous_prior
% (est_is.mod): parameter layout, priors, bounds, dsge_likelihood's fval at a few points,
% set_init_state.m's projection and DsgeSmoother's smoothed initial state (as
% draw_init_state_from_smoother.m calls it) -> ref_is.json.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_is.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare est_is -DMH=0 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, x0, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = struct();
R.names = bayestopt_.name; R.x_init = x0; R.lb = bounds.lb; R.ub = bounds.ub;
R.pshape = bayestopt_.pshape; R.p1 = bayestopt_.p1; R.p2 = bayestopt_.p2; R.p3 = bayestopt_.p3;
R.p4 = bayestopt_.p4; R.p6 = bayestopt_.p6; R.p7 = bayestopt_.p7;
R.state_var = oo_.dr.state_var; R.lik_init = options_.lik_init; R.Harvey = options_.Harvey_scale_factor;
IB = startsWith(bayestopt_.name, 'init ');
pts = {x0};
x = x0; x(IB) = x0(IB) + [0.01; -0.02; 0.015; 0.03; -0.01];  pts{end+1} = x;   % violates s = 2a + b/2
x = x0; x(~IB) = x0(~IB).*(1 + 0.02*[1;-1;1;-1;1;-1;1]); pts{end+1} = x;
x = pts{2}; x(~IB) = pts{3}(~IB); pts{end+1} = x;
R.points = {};
for k = 1:numel(pts)
    x = pts{k};
    P = struct('x', x);
    [P.fval, info] = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    P.info = info(1);
    M1 = set_all_parameters(x, estim_params_, M_);
    ys0 = evaluate_steady_state(oo_.steady_state, [oo_.exo_steady_state; oo_.exo_det_steady_state], M1, options_, true);
    [xc, icheck] = set_init_state(x, ys0, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    P.set_init_state = xc; P.icheck = icheck;
    % ys0 from another point (the slice sampler's coordinate start)
    M2 = set_all_parameters(pts{1}, estim_params_, M_);
    ys1 = evaluate_steady_state(oo_.steady_state, [oo_.exo_steady_state; oo_.exo_det_steady_state], M2, options_, true);
    [xc1, icheck1] = set_init_state(x, ys1, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    P.set_init_state_ys1 = xc1; P.icheck_ys1 = icheck1;
    % DsgeSmoother as in draw_init_state_from_smoother.m
    o = options_; o.lik_init = 1; o.smoother_redux = true; o.smoothed_state_uncertainty = true; o.noprint = true;
    Mx = set_all_parameters(x, estim_params_, M_); Mx.endo_initial_state.status = false;
    [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,alphahat0,state_uncertainty0] = DsgeSmoother(x, dataset_.nobs, transpose(dataset_.data), ...
        dataset_info.missing.aindex, dataset_info.missing.state, Mx, oo_.dr, oo_.steady_state, oo_.exo_steady_state, ...
        oo_.exo_det_steady_state, o, bayestopt_, estim_params_);
    rv = oo_.dr.restrict_var_list(bayestopt_.mf0);
    P.alphahat0 = alphahat0(rv); P.V0 = state_uncertainty0(rv, rv);
    R.points{end+1} = P;
end
R.mf0 = bayestopt_.mf0; R.restrict_var_list = oo_.dr.restrict_var_list;
fid = fopen('ref_is.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
% estimated_params_init(use_calibration) with the initial states (Dynare >= 7.2; 7.1 crashed in
% do_parameter_initialization.m) -> ref_is_usecal.json
cd(here);
dynare est_is -DMH=0 -DUSECAL nointeractive
[~, ~, x0, ~, M_, options_, oo_, estim_params_, bayestopt_] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = struct('names', {bayestopt_.name}, 'x_init', x0);
fid = fopen('ref_is_usecal.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
