% dump_misc.m -- Dynare (7.2 by default, see DYNARE_VERSION) reference values for stream X (reference/misc):
%   x_lik.mod: dsge_likelihood at 3 points and the smoothed variables for lik_init 1/2/4/5,
%     observation_trends (with/without prefilter), a unit-root variant, the .m datafile;
%     the bounds after estimated_params_bounds / estimated_params_remove.
%   x_misc.mod: console output of resid / model_info / prior table, the file written by
%     save_params_and_steady_state, oo_.var after the parameter changes.
%   schorfheide_2000_data.m: the series gy_obs, gp_obs it defines.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_misc.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
global M_ options_ oo_ estim_params_ bayestopt_
runs = {{'LIKINIT=1', '-DLIKINIT=1'}, {'LIKINIT=2', '-DLIKINIT=2'}, {'LIKINIT=4', '-DLIKINIT=4'}, ...
        {'LIKINIT=5', '-DLIKINIT=5'}, {'TREND', '-DTREND'}, {'TREND_PREFILTER', '-DTREND -DPREFILTER'}, ...
        {'UNITROOT_LIKINIT=2', '-DUNITROOT -DLIKINIT=2'}, {'UNITROOT_LIKINIT=4', '-DUNITROOT -DLIKINIT=4'}, ...
        {'UNITROOT_LIKINIT=5', '-DUNITROOT -DLIKINIT=5'}, {'MFILE', '-DMFILE'}};
out = struct();
out.runs = {};
for r = 1:numel(runs)
    R = struct('name', runs{r}{1}, 'defines', runs{r}{2});
    try
        eval(sprintf('dynare x_lik %s nointeractive', runs{r}{2}));
        R.SmoothedVariables = oo_.SmoothedVariables;
        [dataset_, dataset_info, xparam_init, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
            dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
        R.xparam_init = xparam_init;
        R.bounds = bounds;
        R.data = dataset_.data;
        pts = {xparam_init, xparam_init .* (1 + 0.05 * sin(1:numel(xparam_init)))', xparam_init * 0.97};
        R.points = {};
        for k = 1:numel(pts)
            [fval, info] = dsge_likelihood(pts{k}, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
            R.points{end+1} = struct('x', pts{k}, 'minus_loglik', fval, 'info', info(1));
        end
        R.error = '';
    catch ME
        R.error = ME.message;
    end
    out.runs{end+1} = R;
end

% x_misc.mod
txt = evalc('dynare x_misc nointeractive');
out.misc_console = txt;
out.misc_var = oo_.var;
out.misc_params = M_.params;
out.misc_ss_file = fileread('x_misc_ss.txt');

% schorfheide_2000_data.m
clear gy_obs gp_obs data_q
run('schorfheide_2000_data.m');
out.schorfheide_gy_obs = gy_obs;
out.schorfheide_gp_obs = gp_obs;

fid = fopen(fullfile(here, 'ref_misc.json'), 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
disp('ref_misc.json written');
