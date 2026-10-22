% dump_gsa_q2.m -- references on nk_q2.mod / nk_ident_q2.mod:
%   CASE 1: irf_calibration / moment_calibration mapped by sensitivity (map_calibration.m):
%           ref_q2_calib.json (the sample, the restriction values and indicators);
%   CASE 2: estimation + identification(parameter_set=posterior_mode): ref_q2_ident_postmode.json;
%   nk_ident_q2: identification(checks_via_subsets=1): ref_q2_ident_subsets.json.
% Logs: q2_nk_c<k>_matlab.log, q2_nk_ident_subsets_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_gsa_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);

dynare nk_q2 -DCASE=1 nointeractive noclearall
copyfile('nk_q2.log', 'q2_nk_c1_matlab.log');
S = load(fullfile('nk_q2', 'gsa', 'nk_q2_prior_restrictions.mat'));
P = load(fullfile('nk_q2', 'gsa', 'nk_q2_prior.mat'));
R = struct();
R.lpmat = P.lpmat; R.lpmat0 = P.lpmat0;
R.xmat = S.xmat; R.irestrictions = S.irestrictions;
R.indx_irf = S.indx_irf; R.indx_moment = S.indx_moment;
R.mat_irf = S.mat_irf; R.mat_moment = S.mat_moment;
R.irf_restrictions = S.endo_prior_restrictions.irf;
R.moment_restrictions = S.endo_prior_restrictions.moment;
fid = fopen('ref_q2_calib.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_calib.json');
clear oo_ M_ options_ estim_params_ bayestopt_

dynare nk_q2 -DCASE=2 nointeractive noclearall
copyfile('nk_q2.log', 'q2_nk_c2_matlab.log');
S = load(fullfile(here, 'nk_q2', 'identification', 'nk_q2_identif.mat'));
R = struct();
R.params_point = S.ide_hess_point.params;
R.xparam1 = oo_.posterior.optimization.mode;
flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_minimal_point', 'ide_reducedform_point', 'ide_dynamic_point'};
for f = flds
    R.(f{1}) = S.(f{1});
end
fid = fopen('ref_q2_ident_postmode.json', 'w'); fprintf(fid, '%s', jsonencode(fullify(R), 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_ident_postmode.json');
clear oo_ M_ options_ estim_params_ bayestopt_

dynare nk_ident_q2 nointeractive noclearall
copyfile('nk_ident_q2.log', 'q2_nk_ident_subsets_matlab.log');
S = load(fullfile(here, 'nk_ident_q2', 'identification', 'nk_ident_q2_identif.mat'));
R = struct();
R.params_point = S.ide_hess_point.params;
flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_minimal_point', 'ide_reducedform_point', 'ide_dynamic_point'};
for f = flds
    R.(f{1}) = S.(f{1});
end
fid = fopen('ref_q2_ident_subsets.json', 'w'); fprintf(fid, '%s', jsonencode(fullify(R), 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_q2_ident_subsets.json');

function x = fullify(x)
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
