% dump_posterior.m -- reference values for the Rust posterior distributions of smoothed
% objects and forecasts (prior_posterior_statistics.m / pm3.m), ref_posterior.json.
%
% Writes sw07_post.mod = sw07.mod with the STAGE 9 estimation turned into a short
% MCMC (mh_replic=2000, 2 chains, sub_draws=200, smoother, filtered_vars, forecast=12), runs
% it, and dumps
%   - the parameter draws prior_posterior_statistics used (metropolis/*_param1.mat, in the
%     order of the stored smoother results), so the Rust side can rerun its smoother on
%     exactly the same draws and match Mean/Median/HPD/deciles to rounding;
%   - oo_.SmoothedVariables, SmoothedShocks, UpdatedVariables, FilteredVariables,
%     MeanForecast (no shock uncertainty) and PointForecast (simulated shocks) summaries.
%
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_posterior.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
src = fileread('sw07.mod');
old = 'mode_file=''sw07/Output/sw07_mode'', mh_replic=0,';
new = 'mode_file=''sw07/Output/sw07_mode'', mh_replic=2000, mh_nblocks=2, mh_jscale=0.25, sub_draws=200,';
assert(contains(src, old));
src = strrep(src, old, new);
fid = fopen('sw07_post.mod', 'w'); fprintf(fid, '%s', src); fclose(fid);
dynare sw07_post -DSTAGE=9 nointeractive
matlab_diary_off = 1;
global M_ options_ oo_ estim_params_ bayestopt_

R = struct();
R.var_list = var_list_;
R.mh_conf_sig = options_.mh_conf_sig;
R.horizon = options_.forecast;
p = load(fullfile(M_.dname, 'metropolis', [M_.fname '_param1.mat']));
R.draws = p.stock;                       % B x npar, order of the stored smoother results
R.param_names = bayestopt_.name;
R.posterior_mean = oo_.posterior_mean.parameters;
f = {'SmoothedVariables', 'SmoothedShocks', 'UpdatedVariables', 'FilteredVariables', 'MeanForecast', 'PointForecast'};
for k = 1:numel(f)
    s = oo_.(f{k});
    R.(f{k}) = struct('Mean', s.Mean, 'Median', s.Median, 'HPDinf', s.HPDinf, 'HPDsup', s.HPDsup, ...
        'deciles', s.deciles);
end

fid = fopen('ref_posterior.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('ref_posterior.json written\n');
