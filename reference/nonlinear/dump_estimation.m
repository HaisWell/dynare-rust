% dump_estimation.m -- full nonlinear estimations of rbcnl.mod with Dynare 7.2 (stream PF):
% 1. -DESTGF: mode of the Gaussian filter (cubature), mode_compute=4 (deterministic, so the
%    mode must agree with dynare-rust's to optimizer tolerance);
% 2. -DESTSIS: RW Metropolis-Hastings (2 x 2000 draws) with the bootstrap particle filter
%    (1000 particles) from that mode (posterior agrees statistically).
% Run from a PRIVATE copy of reference/nonlinear (after dump_nonlinear.m wrote rbcnl_data.csv):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_estimation.m')"
% writes ref_estimation.json.
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
R = struct();
t0 = tic;
dynare rbcnl -DESTGF nointeractive
global M_ oo_ options_ bayestopt_
R.gf = struct('names', {bayestopt_.name}, 'mode', oo_.posterior.optimization.mode, ...
    'mode_sd', oo_.posterior.optimization.Variance, 'log_density_laplace', oo_.MarginalDensity.LaplaceApproximation, ...
    'seconds', toc(t0));
t0 = tic;
dynare rbcnl -DESTSIS nointeractive
global M_ oo_ options_ bayestopt_
names = bayestopt_.name;
pm = zeros(numel(names), 1); hpdl = pm; hpdu = pm; psd = pm;
for k = 1:numel(names)
    n = strrep(names{k}, 'stderr ', '');
    if isfield(oo_.posterior_mean.parameters, n)
        f = 'parameters';
    elseif isfield(oo_.posterior_mean.shocks_std, n)
        f = 'shocks_std';
    else
        f = 'measurement_errors_std';
    end
    pm(k) = oo_.posterior_mean.(f).(n);
    hpdl(k) = oo_.posterior_hpdinf.(f).(n);
    hpdu(k) = oo_.posterior_hpdsup.(f).(n);
    psd(k) = oo_.posterior_std.(f).(n);
end
R.sis = struct('names', {names}, 'posterior_mean', pm, 'hpd_lower', hpdl, 'hpd_upper', hpdu, 'posterior_sd', psd, ...
    'log_density_mhm', oo_.MarginalDensity.ModifiedHarmonicMean, 'seconds', toc(t0));
fid = fopen('ref_estimation.json', 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
disp('dump_estimation: done');
