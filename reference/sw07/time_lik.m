% time_lik.m -- MATLAB + Dynare: one dsge_likelihood evaluation at the mode, 1 thread
% (maxNumCompThreads(1)), best of 3 loops of 300, after `dynare sw07 -DSTAGE=9`.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
cd(fileparts(mfilename('fullpath')));
dynare sw07 -DSTAGE=9 nointeractive
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
mode = load('sw07/Output/sw07_mode.mat');
xm = mode.xparam1(:);
maxNumCompThreads(1);
best = Inf;
for rep = 1:3
    tic;
    for k = 1:300
        dsge_likelihood(xm, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
            oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    end
    best = min(best, toc/300);
end
fprintf('one evaluation at the mode, 1 thread: %.3f ms\n', 1000*best);
