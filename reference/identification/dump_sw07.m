% dump_sw07.m -- identification on the SW07 model (reference for tests/identification.rs),
% written to ref_identification_sw07.json.
%
% Run from a PRIVATE copy of this folder next to a copy of ../sw07/sw07.mod:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_sw07.m')"
%
% The .mod file is sw07.mod up to its estimation call, with the observables that load on
% the unit roots (pinfobs, robs, labobs, pinfexp4obs, pinflrobs) removed from varobs
% (identification needs stationary observables; with any of them Dynare stops with "There are
% NaN's in the theoretical moments"), followed by `identification(parameter_set=calibration,
% diffuse_filter, no_identification_strength)`.
% tests/identification.rs builds the same file (sw07_ident_mod()).
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
dynare_ver = getenv('DYNARE_VERSION'); if isempty(dynare_ver), dynare_ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
txt = fileread('sw07.mod');
cut = strfind(txt, '// ESTIMATION CALL');
txt = txt(1:cut(1)-1);
vo = regexp(txt, '\nvarobs\s', 'once');
head = txt(1:vo-1); tail = txt(vo:end);
for v = {'pinfobs', 'robs', 'labobs', 'pinfexp4obs', 'pinflrobs'}
    % (the last one, `pinflrobs;`, hands its semicolon to the line before)
    tail = regexprep(tail, ['\n\s*' v{1} '\s*(\n|;)'], '$1', 'once');
end
txt = [head tail];
txt = [txt sprintf('\nidentification(parameter_set=calibration, diffuse_filter, no_identification_strength);\n')];
fid = fopen('sw07_ident.mod', 'w'); fprintf(fid, '%s', txt); fclose(fid);
dynare sw07_ident nointeractive nograph
global M_ options_
S = load(fullfile(here, 'sw07_ident', 'identification', 'sw07_ident_identif.mat'));
R = struct();
R.params_point = S.ide_hess_point.params;
R.varobs = options_.varobs;
flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_reducedform_point'};
for f = flds
    R.(f{1}) = S.(f{1});
end
% drop the big Jacobians we do not compare entry by entry
R.ide_reducedform_point = rmfield(R.ide_reducedform_point, {'dREDUCEDFORM', 'si_dREDUCEDFORM', 'norm_dREDUCEDFORM'});
R.ide_moments_point = rmfield(R.ide_moments_point, {'norm_dMOMENTS'});
R.ide_spectrum_point = rmfield(R.ide_spectrum_point, {'norm_dSPECTRUM'});
fid = fopen(fullfile(here, 'ref_identification_sw07.json'), 'w');
fprintf(fid, '%s', jsonencode(fullify(R), 'ConvertInfAndNaN', true));
fclose(fid);

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
