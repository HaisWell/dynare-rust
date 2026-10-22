% dump_mcmc.m -- Dynare 7.2's slice sampler with estimate_initial_states_endogenous_prior on
% est_is.mod (4000 draws, 1 chain): posterior means and standard deviations of the second half
% of the draws (ref_mcmc.json), for a statistical comparison.
% Run from a PRIVATE COPY of this folder:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_mcmc.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare est_is -DMH=4000 nointeractive
global bayestopt_
files = dir(fullfile('est_is', 'metropolis', 'est_is_mh*_blck1.mat'));
X = [];
for k = 1:numel(files)
    d = load(fullfile(files(k).folder, files(k).name));
    X = [X; d.x2]; %#ok<AGROW>
end
X = X(floor(size(X,1)/2)+1:end, :);
R = struct('names', {bayestopt_.name}, 'mean', mean(X)', 'std', std(X)', 'ndraws', size(X,1));
fid = fopen('ref_mcmc.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
