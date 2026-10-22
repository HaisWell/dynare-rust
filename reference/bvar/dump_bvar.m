% dump_bvar.m -- reference values of Dynare's (7.2 by default, see DYNARE_VERSION) BVAR (bvar.density / bvar.toolbox /
% bvar.forecast / bvar.irf) on three columns of the SW07 dataset, written to ref_bvar.json
% (reference for the Rust `bvar` module, tests/bvar.rs).
%
% Run from a PRIVATE copy of reference/bvar containing sw07_obs.csv:
%   cp ../sw07/sw07_obs.csv . ; /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_bvar.m')"
%
% The forecasts and IRFs draw from MATLAB's randn; the dump records the stream of standard
% normals they consume (rng(seed,'twister'); Z = randn(N,1)) so that the Rust port can replay
% exactly the same draws and match to rounding.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare bvar_sw07 nointeractive nograph
global M_ options_ oo_
opt0 = options_;

R = struct();
R.varobs = options_.varobs;

% ---- marginal densities under several priors ---------------------------------------
cfg = { ...
  struct('name', 'defaults', 'maxnlags', 4, 'first_obs', 20, 'nobs', NaN), ...
  struct('name', 'tau_train', 'maxnlags', 4, 'first_obs', 20, 'nobs', NaN, 'bvar_prior_tau', 2.5, 'bvar_prior_train', 10), ...
  struct('name', 'all_options', 'maxnlags', 3, 'first_obs', 15, 'nobs', 70, 'bvar_prior_tau', 1.5, ...
         'bvar_prior_decay', 1.2, 'bvar_prior_lambda', -3, 'bvar_prior_mu', 0, 'bvar_prior_omega', 2, ...
         'bvar_prior_flat', 1, 'bvar_prior_train', 6, 'presample', 2), ...
  struct('name', 'noconstant', 'maxnlags', 3, 'first_obs', 10, 'nobs', 80, 'noconstant', 1, 'bvar_prior_lambda', 2, 'bvar_prior_mu', 3), ...
  struct('name', 'prefilter', 'maxnlags', 2, 'first_obs', 20, 'nobs', NaN, 'prefilter', 1, 'bvar_prior_train', 8) };
R.density = {};
for c = 1:numel(cfg)
    s = cfg{c};
    options_ = opt0;
    options_.bvar_prior_tau = 3; options_.bvar_prior_train = 0; options_.forecast = 0;
    f = fieldnames(s);
    for j = 1:numel(f)
        if ~any(strcmp(f{j}, {'name', 'maxnlags'}))
            options_.(f{j}) = s.(f{j});
        end
    end
    opts_used = options_;
    oo_.bvar = struct();
    bvar.density(s.maxnlags);
    D = s;
    D.log_marginal_data_density = oo_.bvar.log_marginal_data_density;
    % the full posterior/prior of the largest lag
    p = oo_.bvar.posterior{s.maxnlags}; q = oo_.bvar.prior{s.maxnlags};
    D.posterior = struct('df', p.df, 'S', p.S, 'XXi', p.XXi, 'PhiHat', p.PhiHat);
    D.prior = struct('df', q.df, 'S', q.S, 'XXi', q.XXi, 'PhiHat', q.PhiHat);
    D.options = struct('bvar_prior_tau', opts_used.bvar_prior_tau, 'bvar_prior_decay', opts_used.bvar_prior_decay, ...
        'bvar_prior_lambda', opts_used.bvar_prior_lambda, 'bvar_prior_mu', opts_used.bvar_prior_mu, ...
        'bvar_prior_omega', opts_used.bvar_prior_omega, 'bvar_prior_flat', opts_used.bvar_prior_flat, ...
        'bvar_prior_train', opts_used.bvar_prior_train, 'presample', opts_used.presample, ...
        'noconstant', opts_used.noconstant, 'prefilter', opts_used.prefilter);
    R.density{end+1} = D;
end

% ---- forecast ------------------------------------------------------------------------
options_ = opt0;
options_.bvar_prior_tau = 2.5; options_.bvar_prior_train = 10; options_.first_obs = 20;
options_.nobs = 60; options_.forecast = 8; options_.bvar_replic = 200;
nlags = 2;
[ny, nx, posterior] = bvar.toolbox(nlags);
k = ny*nlags + nx;
N = (options_.bvar_replic + 1) * (posterior.df*ny + k*ny + options_.forecast*ny);
rng(7, 'twister'); Z = randn(N, 1);
rng(7, 'twister');
oo_.bvar = struct();
bvar.forecast(nlags);
F = struct('nlags', nlags, 'forecast', options_.forecast, 'bvar_replic', options_.bvar_replic, ...
    'first_obs', 20, 'nobs', 60, 'bvar_prior_tau', 2.5, 'bvar_prior_train', 10, 'conf_sig', options_.bvar.conf_sig);
F.normals = Z;
F.with_shocks = oo_.bvar.forecast.with_shocks;
F.no_shock = oo_.bvar.forecast.no_shock;
F.rmse = oo_.bvar.forecast.rmse;
R.forecast = F;

% ---- IRFs (both identifications) -------------------------------------------------------
options_.irf = 12; options_.bvar_replic = 150;
N = options_.bvar_replic * (posterior.df*ny + k*ny);
R.irf = {};
for id = {'Cholesky', 'SquareRoot'}
    rng(11, 'twister'); Z = randn(N, 1);
    rng(11, 'twister');
    oo_.bvar = struct();
    bvar.irf(nlags, id{1});
    I = struct('identification', id{1}, 'nlags', nlags, 'irf', options_.irf, 'bvar_replic', options_.bvar_replic);
    I.normals = Z;
    I.Mean = oo_.bvar.irf.Mean; I.Median = oo_.bvar.irf.Median; I.Var = oo_.bvar.irf.Var;
    I.Upper_bound = oo_.bvar.irf.Upper_bound; I.Lower_bound = oo_.bvar.irf.Lower_bound;
    R.irf{end+1} = I;
end

fid = fopen('ref_bvar.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('ref_bvar.json written');
