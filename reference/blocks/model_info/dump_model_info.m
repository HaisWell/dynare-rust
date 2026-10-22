% dump_model_info.m -- the text of model_info(block_static|block_dynamic[, incidence]) under
% MATLAB, cut from the first "Information about" header to the last legend line:
% <name>.txt for every mi_*.mod of this folder, plus mi_sw07 / mi_qpm / mi_gm05 (the model of
% ../../{sw07,qpm,gm05} up to varobs, then the model_info commands and a stoch_simul, as in
% tests/model_info_blocks.rs). Run from a PRIVATE copy of reference/ (Dynare writes next to
% the .mod):
%   cd blocks/model_info; /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_model_info.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
cmds = sprintf(['\nmodel_info(block_static);\nmodel_info(block_static, incidence);\n' ...
    'model_info(block_dynamic);\nmodel_info(block_dynamic, incidence);\n' ...
    'stoch_simul(order=1, irf=0, noprint, nomoments, nocorr, nofunctions, nograph);\n']);
semi = {'mi_sw07', '../../sw07/sw07.mod'; 'mi_qpm', '../../qpm/model_qpm.mod'; 'mi_gm05', '../../gm05/model_gm05.mod'};
for k = 1:size(semi, 1)
    src = fileread(semi{k, 2});
    cut = strfind(src, sprintf('\nvarobs'));
    fid = fopen([semi{k, 1} '.mod'], 'w'); fprintf(fid, '%s', [src(1:cut(1)) newline cmds]); fclose(fid);
end
d = dir('mi_*.mod');
names = cellfun(@(s) s(1:end-4), {d.name}, 'UniformOutput', false);
endl = '1: non-null element, X: non-null element related to a state variable';
for k = 1:numel(names)
    nm = names{k};
    txt = evalc(sprintf('dynare %s nointeractive nograph console', nm));
    lines = strsplit(txt, newline, 'CollapseDelimiters', false);
    a = find(startsWith(lines, 'Information about'), 1);
    b = find(strcmp(lines, endl), 1, 'last');
    fid = fopen([nm '.txt'], 'w'); fprintf(fid, '%s\n', strjoin(lines(a:b), newline)); fclose(fid);
    fprintf('%s: lines %d-%d\n', nm, a, b);
end
for k = 1:size(semi, 1), delete([semi{k, 1} '.mod']); end
