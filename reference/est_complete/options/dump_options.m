% dump_options.m -- Dynare reference values (DYNARE_VERSION, default 7.2) for the estimation options of the
% estimation-completion stream (tests/est_complete_options.rs):
%   est_opts.mod STAGE 1-3 (dr=cycle_reduction, dr=logarithmic_reduction, qz_zero_threshold):
%     dsge_likelihood at three points; STAGE 4/5/7: initial_estimation_checks errors;
%     STAGE 6 (no_init_estimation_check_first_obs): fval at the initial point;
%     STAGE 8 (frequentist_smoother without smoother): oo_.SmoothedVariables;
%   posterior_moments.m's kernel density on a fixed vector;
%   nonlinear_filter_initialization = 3 (options_.particle.initialization) with the
%     deterministic nonlinear filters on ../../nonlinear/rbcnl.mod (order 2).
% Run from a PRIVATE COPY of reference/est_complete/options with ../../nonlinear next to it:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_options.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
R = struct();
for STAGE = 1:8
    clear global
    global M_ options_ oo_ estim_params_ bayestopt_
    msg = '';
    try
        eval(sprintf('dynare est_opts -DSTAGE=%d nointeractive', STAGE));
    catch e
        msg = e.message;
        disp(msg);
    end
    S = struct('error', msg);
    if ismember(STAGE, [1 2 3 6])
        [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
            dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
        S.xparam_init = xparam_init;
        S.dr_cycle_reduction = options_.dr_cycle_reduction;
        S.dr_logarithmic_reduction = options_.dr_logarithmic_reduction;
        S.qz_zero_threshold = options_.qz_zero_threshold;
        pts = {xparam_init, xparam_init .* [1.02; 0.97; 1.1; ones(numel(xparam_init) - 3, 1)], xparam_init * 0.95};
        S.points = {};
        for k = 1:numel(pts)
            [fval, info] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            S.points{end+1} = struct('x', pts{k}, 'fval', fval, 'info', info(1));
        end
    end
    if STAGE == 8
        S.SmoothedVariables = oo_.SmoothedVariables;
        S.SmoothedShocks = oo_.SmoothedShocks;
    end
    R.(sprintf('stage%d', STAGE)) = S;
end

% ---- posterior_moments.m: kernel density -------------------------------------------
xx = sin((1:1500)' * 0.7) .* (1 + 0.3 * cos((1:1500)' * 0.13)) + 0.05 * ((1:1500)' / 1500);
[pm, pmed, pv, hpd, dec, dens] = posterior_moments(xx, 0.9);
R.kde = struct('xx', xx, 'mean', pm, 'median', pmed, 'var', pv, 'hpd', hpd, 'deciles', dec, 'density', dens);
[~, ~, ~, ~, ~, dens0] = posterior_moments(ones(50, 1), 0.9);
R.kde_constant = dens0;

% ---- nonlinear_filter_initialization = 3 ---------------------------------------------
cd(fullfile(here, '..', '..', 'nonlinear'));
clear global
global M_ options_ oo_ estim_params_ bayestopt_
try
    dynare rbcnl -DORDER=2 nointeractive
catch e
    disp(e.message);
end
options_.particle = check_particle_filter_options(options_.particle);
[dataset_, dataset_info, xparam1, ~, M_, options_, oo_, estim_params_, bayestopt_, BoundsInfo] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
x0 = xparam1; x1 = x0 .* [0.98; 1.1; 1.2; 0.9; 1.15];
N = struct('x0', x0, 'x1', x1, 'cases', {{}});
C = {{'gf_cubature', 'gf', 'cubature'}, {'nlkf_unscented', 'nlkf', 'unscented'}};
for k = 1:numel(C)
    o = options_;
    o.particle.initialization = 3;
    o.particle.filter_algorithm = C{k}{2};
    o.particle.proposal_approximation.cubature = false; o.particle.proposal_approximation.unscented = false;
    o.particle.proposal_approximation.montecarlo = false;
    o.particle.proposal_approximation.(C{k}{3}) = true;
    o.particle.distribution_approximation.cubature = false; o.particle.distribution_approximation.unscented = false;
    o.particle.distribution_approximation.montecarlo = false;
    o.particle.distribution_approximation.(C{k}{3}) = true;
    o.particle = check_particle_filter_options(o.particle);
    f = zeros(2, 1);
    xs = {x0, x1};
    for ip = 1:2
        f(ip) = non_linear_dsge_likelihood(xs{ip}, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, ...
            BoundsInfo, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    end
    N.cases{end+1} = struct('name', C{k}{1}, 'algorithm', C{k}{2}, 'approximation', C{k}{3}, ...
        'initial_state_prior_std', o.particle.initial_state_prior_std, 'fval', f);
    fprintf('dump_options: initialization 3 %-16s fval = %.10f %.10f\n', C{k}{1}, f(1), f(2));
end
R.nonlinear_init3 = N;
cd(here);
fid = fopen('ref_options.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('dump_options: done');
