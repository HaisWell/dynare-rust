% dump_mode_check_schur.m -- two mode_check points with MATLAB's Schur basis of T.
% Rationale:
% T has a multiple unit root, and which of its Schur vectors compute_Pinf_Pstar drops as not
% moving the observables (norm(QT(mf,:)*ST(:,k)) < 1e-8) depends on the LAPACK build. SW07 also
% has a double unit root (pistar, labtrend); the points below (crhoa z(15), cprobw z(6)) keep
% the schema and the check, pick other points if a mode_check comparison flags any. Dumps, per
% point, x, fval and MATLAB's (unordered) real Schur form of T(restrict_columns,
% restrict_columns), so that the mode_check test can run dynare-rust's filter on Dynare's basis.
% Run from a private copy of reference/sw07 (with ref_mode_check.json):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_mode_check_schur.m')"
% Writes ref_mode_check_schur.json.
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare sw07 -DSTAGE=9 nointeractive nograph
cd(here);
global M_ options_ oo_ estim_params_ bayestopt_
[dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
R = jsondecode(fileread('ref_mode_check.json'));
pts = {'crhoa', 15; 'cprobw', 6};
out = cell(size(pts, 1), 1);
for i = 1:size(pts, 1)
    k = find(strcmp(R.names, pts{i,1}));
    if iscell(R.params), p = R.params{k}; else, p = R.params(k); end
    x = R.xparam1; x(k) = p.z(pts{i,2});
    fval = dsge_likelihood(x, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    M2 = set_all_parameters(x, estim_params_, M_);
    [T, Rm, SS, ~, dr] = dynare_resolve(M2, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 'restrict');
    [Pstar, Pinf] = compute_Pinf_Pstar(bayestopt_.mf1, T, Rm, M2.Sigma_e, options_.qz_criterium, dr.restrict_columns);
    indx = dr.restrict_columns;
    [QT0, ST0] = schur(T(indx, indx));
    lnprior = priordens(x, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
    out{i} = struct('name', pts{i,1}, 'index', k, 'z_index', pts{i,2}, 'x', x, 'fval', fval, 'lnprior', lnprior, ...
        'trace_Pinf', trace(Pinf), 'schur_U', QT0, 'schur_T', ST0, 'schur_wr', real(ordeig(ST0)), 'schur_wi', imag(ordeig(ST0)));
    fprintf('%s z=%.6f fval=%.10f trace(Pinf)=%g\n', pts{i,1}, x(k), fval, trace(Pinf));
end
fid = fopen('ref_mode_check_schur.json', 'w'); fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true)); fclose(fid);
