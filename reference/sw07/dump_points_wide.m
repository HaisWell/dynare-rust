% dump_points_wide.m -- Dynare's dsge_likelihood fval at the points in points_wide.json, bounds
% switched off (read by tests/dsge_sw07.rs, sw07_posterior_kernel_wide);
% if points_wide.json does not exist yet, this script writes 40 points (8 on a
% line along stderr ea, then draws from N(mode, 4*inv(hh)) with a finite prior, rng(7)), in
% the same schema ({"x": [...]}, without "rust_fval").
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
cd(fileparts(mfilename('fullpath')));
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
np = numel(bayestopt_.name);
if ~exist('points_wide.json', 'file')
    mode = load('sw07/Output/sw07_mode.mat');
    xm = mode.xparam1(:);
    k = find(strcmp(bayestopt_.name, 'SE_ea'));
    X = [];
    for v = [0.2 0.3 0.4 0.5 0.6 0.7 0.85 1.0]
        x = xm; x(k) = v; X(end+1, :) = x'; %#ok<SAGROW>
    end
    C = chol(inv(mode.hh), 'lower');
    rng(7);
    while size(X, 1) < 40
        x = xm + 2*C*randn(np, 1);
        if isfinite(priordens(x, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4))
            X(end+1, :) = x'; %#ok<SAGROW>
        end
    end
    fid = fopen('points_wide.json', 'w'); fprintf(fid, '%s', jsonencode(struct('x', X))); fclose(fid);
end
bounds.lb(:) = -Inf; bounds.ub(:) = Inf;
P = jsondecode(fileread('points_wide.json'));
X = P.x; if size(X,2) ~= np, X = X'; end
R = struct('fval', zeros(size(X,1),1), 'info', zeros(size(X,1),1));
for k = 1:size(X,1)
    [f, info] = dsge_likelihood(X(k,:)', dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    R.fval(k) = f; R.info(k) = info(1);
end
fid = fopen('ref_points_wide.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
