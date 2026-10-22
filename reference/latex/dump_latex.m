% dump_latex.m -- LaTeX outputs and model_comparison of Dynare (7.2 by default, see DYNARE_VERSION) on latex_nk.mod /
% latex_nk_alt.mod (reference for src/latex and src/estimation/model_comparison.rs).
% Run from a PRIVATE copy of reference/latex:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_latex.m')"
% The .tex files are copied to reference/latex/expected/ by hand afterwards.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end % DYNARE_VERSION=7.1 for the 7.1 baseline
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare latex_nk_alt nointeractive
alt_laplace = oo_.MarginalDensity.LaplaceApproximation;
alt_mode = oo_.posterior_mode;
dynare latex_nk nointeractive
R = struct();
R.laplace = struct('latex_nk', oo_.MarginalDensity.LaplaceApproximation, 'latex_nk_alt', alt_laplace);
R.model_comparison = oo_.Model_Comparison;
R.mode = oo_.posterior_mode;
R.alt_mode = alt_mode;
fid = fopen('ref_model_comparison.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
