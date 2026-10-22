% dump_identification.m -- reference values for tests/identification.rs, written to
% ref_identification_<case>.json (CASE 1 and 2 of nk_ident.mod, and nk_noprior.mod).
%
% Run from a PRIVATE copy of this folder (Dynare writes +nk_ident/ and nk_ident/ next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_identification.m')"
%
% For each case: runs `dynare nk_ident -DCASE=k`, loads nk_ident/identification/nk_ident_identif.mat
% (the point analysis of identification.run) and dumps the Jacobians, the checks and, for
% CASE 1, the identification strength together with the simulated data it was computed from
% (re-simulated with the default seed, as identification.analysis -> stoch_simul does it).
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
% {mod file, defines, output suffix, dump the simulated sample}
runs = {{'nk_ident', '-DCASE=1', '1', true}, {'nk_ident', '-DCASE=2', '2', false}, ...
        {'nk_noprior', '', 'noprior', true}};
for irun = 1:numel(runs)
    clearvars -except here irun runs
    close all
    run_ = runs{irun};
    eval(sprintf('dynare %s %s nointeractive nograph', run_{1}, run_{2}));
    global M_ options_ oo_ estim_params_ bayestopt_
    S = load(fullfile(here, run_{1}, 'identification', [run_{1} '_identif.mat']));
    R = struct();
    R.params_point = S.ide_hess_point.params;
    R.M_params = M_.params;
    flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_minimal_point', 'ide_reducedform_point', 'ide_dynamic_point', 'ide_hess_point'};
    for f = flds
        R.(f{1}) = S.(f{1});
    end
    R.options_ident = S.store_options_ident;
    if run_{4}
        % Re-create the simulated data of the identification strength analysis.
        params = S.ide_hess_point.params;
        % parameter order: stderr eg eu er, then the parameters in declaration order
        M2 = M_; M2.Sigma_e = diag(params(1:3).^2); M2.params = params(4:end)';
        opts = options_;
        opts.order = 1; opts.irf = 0; opts.noprint = 1; opts.SpectralDensity.trigger = 0;
        opts.periods = 300 + 100; opts.nomoments = 0; opts.ar = 1;
        set_dynare_seed('default');
        [info, oo2] = stoch_simul(M2, opts, oo_, opts.varobs);
        R.sim_info = info(1);
        R.endo_simul = oo2.endo_simul;
        R.varobs_id = opts.varobs_id;
        R.sim_ys = oo2.dr.ys;
        R.order_var = oo2.dr.order_var;
    end
    fid = fopen(fullfile(here, ['ref_identification_' run_{3} '.json']), 'w');
    fprintf(fid, '%s', jsonencode(fullify(R), 'ConvertInfAndNaN', true));
    fclose(fid);
end

function x = fullify(x)
% sparse -> full, recursively (jsonencode refuses sparse matrices)
if issparse(x)
    x = full(x);
elseif isstruct(x)
    for k = 1:numel(x)
        f = fieldnames(x);
        for i = 1:numel(f)
            x(k).(f{i}) = fullify(x(k).(f{i}));
        end
    end
elseif iscell(x)
    for k = 1:numel(x)
        x{k} = fullify(x{k});
    end
end
end
