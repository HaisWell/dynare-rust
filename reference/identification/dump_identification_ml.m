% dump_identification_ml.m -- identification of nk_ml.mod (estimated_params without priors,
% i.e. maximum likelihood): Dynare tests the "ML_Starting_value" point whatever parameter_set
% says and ignores prior_mc. Writes ref_identification_ml.json (same fields as
% dump_identification.m, plus the printed text) for tests/identification.rs.
% Run from a PRIVATE copy of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_identification_ml.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk_ml nointeractive nograph
global M_ options_ oo_ estim_params_ bayestopt_
S = load(fullfile(here, 'nk_ml', 'identification', 'nk_ml_identif.mat'));
R = struct();
R.params_point = S.ide_hess_point.params;
R.files = {dir(fullfile(here, 'nk_ml', 'identification', '*.mat')).name};
flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_minimal_point', 'ide_reducedform_point', 'ide_dynamic_point', 'ide_hess_point'};
for f = flds
    R.(f{1}) = full_(S.(f{1}));
end
R.options_ident = S.store_options_ident;
fid = fopen(fullfile(here, 'ref_identification_ml.json'), 'w');
fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
fclose(fid);

function x = full_(x)
% sparse -> full, recursively (jsonencode refuses sparse matrices)
if issparse(x)
    x = full(x);
elseif isstruct(x)
    for k = 1:numel(x)
        f = fieldnames(x);
        for i = 1:numel(f)
            x(k).(f{i}) = full_(x(k).(f{i}));
        end
    end
elseif iscell(x)
    for k = 1:numel(x)
        x{k} = full_(x{k});
    end
end
end
