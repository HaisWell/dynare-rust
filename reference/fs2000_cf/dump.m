% Reference for tests/fs2000_calib_smoother_cf.rs (Dynare 7.2). randn is shadowed by zeros so
% the conditional forecast is deterministic (Mean = the zero-shock path).
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath')); cd(here);
dynare fs2000_sim nointeractive
ip = strmatch('gp_obs', M_.endo_names, 'exact'); iy = strmatch('gy_obs', M_.endo_names, 'exact');
Y = oo_.endo_simul([ip iy], :);
fid = fopen('fs2000_cf_data.csv', 'w'); fprintf(fid, 'gp_obs,gy_obs\n');
fprintf(fid, '%.17g,%.17g\n', Y(:, 1:120)); fclose(fid);
clear M_ oo_ options_
zr = fullfile(tempdir, sprintf('zero_randn_%d', feature('getpid'))); mkdir(zr);
fid = fopen(fullfile(zr, 'randn.m'), 'w'); fprintf(fid, 'function x = randn(varargin)\nx = zeros(varargin{:});\n'); fclose(fid);
addpath(zr, '-begin');
dynare fs2000_calib_smoother_cf nointeractive
rmpath(zr);
R.endo_names = M_.endo_names;
R.cond_mean = oo_.conditional_forecast.cond.Mean;
R.uncond_mean = oo_.conditional_forecast.uncond.Mean;
R.controlled_exo = oo_.conditional_forecast.controlled_exo_variables.Mean;
for v = {'m','P','c','y'}
  R.smoothed.(v{1}) = oo_.SmoothedVariables.(v{1});
end
fid = fopen('ref.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
