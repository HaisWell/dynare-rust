% dump_prior_trunc.m -- prior_bounds(bayestopt_, 1e-10) for the 40 SW07 priors (the bayestopt
% of ref.json, so run dump_reference.m first) and chi2inv at 40 degrees of freedom. Writes
% ref_prior_trunc.json, read by tests/estimation_prior.rs.
% Run from a private copy of reference/sw07:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_prior_trunc.m')"
% Dynare version: set DYNARE_VERSION (e.g. 7.1) in the environment to dump another release.
dynare_version = getenv('DYNARE_VERSION'); if isempty(dynare_version), dynare_version = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', dynare_version); end
addpath(fullfile(droot, 'matlab'));
cd(fileparts(mfilename('fullpath')));
dynare_config;
R = jsondecode(fileread('ref.json'));
bp = R.bayestopt;
f = {'p3', 'p4'};
for k = 1:2
    v = bp.(f{k});  % jsonencode writes -Inf/Inf as null: a cell array when some are missing
    if iscell(v), w = NaN(numel(v), 1); for j = 1:numel(v), if ~isempty(v{j}), w(j) = v{j}; end, end, bp.(f{k}) = w; end
end
bp.p3(isnan(bp.p3)) = -Inf; bp.p4(isnan(bp.p4)) = Inf;
np = numel(bp.pshape);
bb = prior_bounds(bp, 1e-10);
out = struct('prior_trunc', 1e-10, 'lb', bb.lb, 'ub', bb.ub, 'chi2_dof', np, ...
    'chi2_p', [0.1 0.5 0.9], 'chi2', [chi2inv(0.1, np) chi2inv(0.5, np) chi2inv(0.9, np)]);
fid = fopen('ref_prior_trunc.json', 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
fprintf('%d priors, chi2inv(0.5, %d) = %.10f\n', np, np, out.chi2(2));
