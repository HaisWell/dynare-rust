% dump_shocks.m -- shocks / shock_paths reference for dynare-rust (stream shocks/shock_paths).
% Writes ref_shocks.json: for each case oo_.endo_simul, oo_.exo_simul, oo_.steady_state,
% oo_.exo_steady_state (and oo_.pfwee.terminal_info / shocks_info with expectation errors),
% M_.Sigma_e / M_.H of the stochastic case, and closed skew normal draws
% (rand_multivariate_csn.m) with the standard normals they consumed.
%
% Run from a PRIVATE copy of reference/shocks:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_shocks.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
cases = {
    'det1', 'sh_det', '-DCASE=1'
    'det2', 'sh_det', '-DCASE=2'
    'det3', 'sh_det', '-DCASE=3'
    'det4', 'sh_det', '-DCASE=4'
    'sp1',  'sh_sp',  '-DCASE=1'
    'sp2',  'sh_sp',  '-DCASE=2'
    'sp3',  'sh_sp',  '-DCASE=3'
    'stoch1', 'sh_stoch', '-DCASE=1'
    'stoch2', 'sh_stoch', '-DCASE=2'
};
global M_ options_ oo_
R = struct();
for c = 1:size(cases, 1)
    name = cases{c, 1};
    S = struct();
    try
        eval(sprintf('dynare %s %s nointeractive nograph noclearall console', cases{c, 2}, cases{c, 3}));
        S.endo_names = M_.endo_names;
        S.steady_state = oo_.steady_state;
        S.exo_steady_state = oo_.exo_steady_state;
        S.Sigma_e = M_.Sigma_e;
        S.H = M_.H;
        if isfield(oo_, 'endo_simul') && ~isempty(oo_.endo_simul)
            S.endo_simul = oo_.endo_simul;
            S.exo_simul = oo_.exo_simul;
        end
        if isfield(oo_, 'pfwee')
            S.terminal_info = oo_.pfwee.terminal_info;
            S.shocks_info = oo_.pfwee.shocks_info;
        end
    catch err
        S.failure = getReport(err);
        disp(S.failure);
    end
    R.(name) = S;
    clear -global oo_
    global M_ options_ oo_
end

% closed skew normal draws as simult.m makes them (rand_multivariate_csn), with the normals
csn = csn_update_specification(diag([0.01^2, 0.02^2, 0.005^2]), [1 1 1 0.6; 3 3 3 -0.4]);
rng(7);
Z = randn(1, 40000);
rng(7);
X = rand_multivariate_csn(1000, csn.mu_e, csn.Sigma_e, csn.Gamma_e, csn.nu_e, csn.Delta_e);
R.csn.mu = csn.mu_e;
R.csn.Sigma = csn.Sigma_e;
R.csn.Gamma = csn.Gamma_e;
R.csn.normals = Z;
R.csn.draws = X;
rng(8);
X = rand_multivariate_csn(100000, csn.mu_e, csn.Sigma_e, csn.Gamma_e, csn.nu_e, csn.Delta_e);
R.csn.mean = mean(X, 2);
R.csn.var = var(X, 0, 2);
R.csn.skewness = skewness(X, 1, 2);

fid = fopen('ref_shocks.json', 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);
