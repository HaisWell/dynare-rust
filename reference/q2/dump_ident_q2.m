% dump_ident_q2.m -- identification at order 2/3 (analytic_derivation_mode = -1
% and the default analytic mode) and checks_via_subsets on rbc_q2.mod (CASE 10..13): the point
% analysis of <fname>_identif.mat to ref_q2_ident_c<k>.json, the logs to q2_c<k>_matlab.log.
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_ident_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
for cs = [10 11 12 13]
    eval(sprintf('dynare rbc_q2 -DCASE=%d nointeractive noclearall', cs));
    copyfile('rbc_q2.log', sprintf('q2_c%d_matlab.log', cs));
    S = load(fullfile(here, 'rbc_q2', 'identification', 'rbc_q2_identif.mat'));
    R = struct();
    R.params_point = S.ide_hess_point.params;
    flds = {'ide_moments_point', 'ide_spectrum_point', 'ide_minimal_point', 'ide_reducedform_point', 'ide_dynamic_point'};
    for f = flds
        R.(f{1}) = S.(f{1});
    end
    fid = fopen(fullfile(here, sprintf('ref_q2_ident_c%d.json', cs)), 'w');
    fprintf(fid, '%s', jsonencode(fullify(R), 'ConvertInfAndNaN', true));
    fclose(fid);
    fprintf('wrote ref_q2_ident_c%d.json\n', cs);
end

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
