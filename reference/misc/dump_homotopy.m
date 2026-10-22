% dump_homotopy.m -- Dynare (7.2 by default, see DYNARE_VERSION) reference for steady(homotopy_mode=1|2|3) on x_homotopy.mod
% (steady state, parameters, exogenous steady state, console output) -> ref_homotopy.json.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_homotopy.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
global M_ options_ oo_
out = struct();
out.runs = {};
for mode = 1:3
    txt = evalc(sprintf('dynare x_homotopy -DMODE=%d nointeractive', mode));
    out.runs{end+1} = struct('mode', mode, 'console', txt, 'steady_state', oo_.steady_state, ...
        'params', M_.params, 'exo_steady_state', oo_.exo_steady_state);
end
fid = fopen(fullfile(here, 'ref_homotopy.json'), 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
disp('ref_homotopy.json written');
