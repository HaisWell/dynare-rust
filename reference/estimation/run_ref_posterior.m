% run_ref_posterior.m -- reference values for the posterior statistics tests
% (mcmc_ifac.m, posterior_moments.m) on deterministic series; writes ml_ref_posterior.json.
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run_ref_posterior"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
dynare_config;
t = (1:5001)';
x = sin(0.1*t) + 0.5*cos(0.37*t.^1.1) + 0.01*mod(t.^2, 17);
out = struct();
out.ifac_1000 = mcmc_ifac(x, 1000);
out.ifac_2500_5 = mcmc_ifac(x, 2500.5);
out.ifac_short = mcmc_ifac(x(1:333), 1000);
[m, med, v, hpd, dec] = posterior_moments(x, 0.9);
out.pm = struct('mean', m, 'median', med, 'var', v, 'hpd', hpd, 'deciles', dec);
[m, med, v, hpd, dec] = posterior_moments(x(1:2000), 0.8);
out.pm2 = struct('mean', m, 'median', med, 'var', v, 'hpd', hpd, 'deciles', dec);
fid = fopen('ml_ref_posterior.json', 'w'); fprintf(fid, '%s', jsonencode(out)); fclose(fid);
