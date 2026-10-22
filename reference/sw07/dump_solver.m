% dump_solver.m -- steady state + first-order decision rule + restricted state
% space of Dynare at several parameter vectors, written to ref_solver.json
% (reference for the Rust `solver` module).
%
% Run from a PRIVATE copy of reference/sw07 (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_solver.m')"
%
% Mimics dsge_likelihood -> dynare_resolve(...,'restrict') -> resol ->
% evaluate_steady_state / stochastic_solvers at each point, from two initial
% guesses: the calibrated steady state (what `steady` leaves in oo_ before the
% estimation command) and the steady state oo_ holds after the STAGE=9 run.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);

% ---- 1. calibrated steady state: the .mod file without the estimation call
txt = fileread('sw07.mod');
cut = strfind(txt, '// ESTIMATION CALL');
fid = fopen('model_ss.mod', 'w'); fprintf(fid, '%s', txt(1:cut(1)-1)); fclose(fid);
dynare model_ss nointeractive
ys_calib = oo_.steady_state; params_calib = M_.params; %#ok<NASGU>
save('calib_tmp.mat', 'ys_calib', 'params_calib');

% ---- 2. the STAGE=9 run (mode, no MCMC), as dump_reference.m
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
load('calib_tmp.mat');
[dataset_, dataset_info, xparam_init, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
ys_mode_run = oo_.steady_state;
exo_ss = oo_.exo_steady_state; exo_det_ss = oo_.exo_det_steady_state;

R = struct();
R.options = struct('solve_algo', options_.solve_algo, 'steady_maxit', options_.steady.maxit, ...
    'solve_tolf', options_.solve_tolf, 'solve_tolx', options_.solve_tolx, 'gstep', options_.gstep(1), ...
    'linear', options_.linear, 'block', options_.block, 'bytecode', options_.bytecode, ...
    'steadystate_flag', options_.steadystate_flag, 'jacobian_flag', options_.jacobian_flag, ...
    'solve_randomize_initial_guess', options_.solve_randomize_initial_guess, ...
    'qz_criterium', options_.qz_criterium, 'qz_zero_threshold', options_.qz_zero_threshold, ...
    'order', options_.order, 'aim_solver', options_.aim_solver, ...
    'dr_cycle_reduction', options_.dr_cycle_reduction, 'loglinear', options_.loglinear);
R.static_and_dynamic_models_differ = M_.static_and_dynamic_models_differ;
R.set_auxiliary_variables = M_.set_auxiliary_variables;
R.ys_calib = ys_calib; R.params_calib = params_calib; R.ys_mode_run = ys_mode_run;

% ---- 3. parameter points
ref = jsondecode(fileread('ref.json'));
pts = {}; names = {};
for k = 1:numel(ref.points)
    pts{end+1} = struct('x', ref.points(k).x(:), 'over', zeros(0, 2)); names{end+1} = sprintf('ref_point_%d', k);
end
xm = ref.xparam_mode(:);
rng(7);
for k = 1:3
    pts{end+1} = struct('x', xm + 0.10*abs(xm).*randn(size(xm)), 'over', zeros(0, 2)); names{end+1} = sprintf('perturb10_%d', k);
end
pid = @(s) find(strcmp(M_.param_names, s));
% steady-state shift (observable constants), an explosive shock process, a negative and a
% too-weak inflation response (Blanchard-Kahn failures)
xs = xm; ip = find(strcmp(ref.estim_param_names, 'constepinf')); xs(ip) = xs(ip) + 0.5;
ip = find(strcmp(ref.estim_param_names, 'constelab')); xs(ip) = xs(ip) - 0.7;
pts{end+1} = struct('x', xs, 'over', zeros(0, 2)); names{end+1} = 'ss_shift';
pts{end+1} = struct('x', xm, 'over', [pid('crhoa'), 1.05]); names{end+1} = 'explosive_crhoa';
pts{end+1} = struct('x', xm, 'over', [pid('crpi'), -3]); names{end+1} = 'taylor_crpi_neg';
pts{end+1} = struct('x', xm, 'over', [pid('crpi'), 0.5]); names{end+1} = 'taylor_crpi_weak';

trip = @(S) struct('i', [], 'j', [], 'v', [], 'm', size(S, 1), 'n', size(S, 2));
R.cases = {};
guesses = {ys_calib, ys_mode_run}; gnames = {'calib', 'mode_run'};
for k = 1:numel(pts)
    for g = 1:2
        M2 = set_all_parameters(pts{k}.x, estim_params_, M_);
        for o = 1:size(pts{k}.over, 1)
            M2.params(pts{k}.over(o, 1)) = pts{k}.over(o, 2);
        end
        guess = guesses{g};
        if M_.set_auxiliary_variables
            guess_aux = feval([M_.fname '.set_auxiliary_variables'], guess, exo_ss, M2.params);
        else
            guess_aux = guess;
        end
        [r0, T_order, Tt] = feval([M_.fname '.static_resid'], guess_aux, exo_ss, M2.params);
        J0 = feval([M_.fname '.static_g1'], guess_aux, exo_ss, M2.params, ...
            M_.static_g1_sparse_rowval, M_.static_g1_sparse_colval, M_.static_g1_sparse_colptr, T_order, Tt);
        [si, sj, sv] = find(sparse(J0));
        [ys, ~, info_ss] = evaluate_steady_state(guess, exo_ss, M2, options_, ~options_.steadystate.nocheck);
        [T, Rm, SS, info, dr] = dynare_resolve(M2, options_, oo_.dr, guess, exo_ss, exo_det_ss, 'restrict');
        C = struct('name', names{k}, 'guess', gnames{g}, 'x', pts{k}.x, 'overrides', pts{k}.over, ...
            'params', M2.params, 'guess_aux', guess_aux, 'static_resid', r0, ...
            'static_g1', struct('i', si, 'j', sj, 'v', sv), 'ys', ys, 'info_ss', info_ss(:)', ...
            'info', info(:)');
        if isfield(dr, 'eigval') && ~isempty(dr.eigval)
            C.eigval_abs = abs(dr.eigval);
        end
        if info(1) == 0
            g1d = feval([M_.fname '.dynamic_g1'], [ys; ys; ys], exo_ss, M2.params, ys, ...
                M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
            [di, dj, dv] = find(g1d);
            C.dynamic_g1 = struct('i', di, 'j', dj, 'v', dv);
            C.ghx = dr.ghx; C.ghu = dr.ghu; C.T = T; C.R = Rm; C.SteadyState = SS;
        else
            % still dump the Jacobian at the attempted steady state (BK failures)
            if info_ss(1) == 0
                g1d = feval([M_.fname '.dynamic_g1'], [ys; ys; ys], exo_ss, M2.params, ys, ...
                    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
                [di, dj, dv] = find(g1d);
                C.dynamic_g1 = struct('i', di, 'j', dj, 'v', dv);
            end
        end
        R.cases{end+1} = C;
        fprintf('%-16s %-9s info=%s info_ss=%s\n', names{k}, gnames{g}, mat2str(info(:)'), mat2str(info_ss(:)'));
    end
end

fid = fopen('ref_solver.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_solver.json written\n');
