% dump_stoch_simul.m -- run Dynare 7.2 on nk.mod (check + five stoch_simul commands) and
% write what each command leaves in oo_ to ref_nk.json (reference for src/analysis).
% The console output goes to nk.log (Dynare's diary), kept as nk_matlab.log for comparing
% the printed tables.
%
% Run from a PRIVATE copy of reference/analysis (Dynare writes next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_stoch_simul.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare nk -DMATLAB_DUMP nointeractive
copyfile('nk.log', 'nk_matlab.log');

R = struct();
R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.Sigma_e = M_.Sigma_e;
R.steady_state = oo_.steady_state;
cmds = {};
for k = 1:4
    S = load(sprintf('nk_cmd%d.mat', k));
    o = S.oo_; op = S.options_;
    C = struct();
    C.var_list = o.var_list;
    C.ghx = o.dr.ghx; C.ghu = o.dr.ghu; C.order_var = o.dr.order_var; C.ys = o.dr.ys;
    C.eigval_re = real(o.dr.eigval); C.eigval_im = imag(o.dr.eigval);
    C.hp_filter = op.hp_filter; C.ar = op.ar; C.irf = op.irf; C.periods = op.periods; C.drop = op.drop;
    if isfield(o, 'irfs'), C.irfs = o.irfs; end
    if isfield(o, 'gamma_y'), C.gamma_y = o.gamma_y; end
    if isfield(o, 'mean'), C.mean = o.mean; end
    if isfield(o, 'var'), C.var = o.var; end
    if isfield(o, 'autocorr'), C.autocorr = o.autocorr; end
    if isfield(o, 'skewness'), C.skewness = o.skewness; end
    if isfield(o, 'kurtosis'), C.kurtosis = o.kurtosis; end
    if isfield(o, 'variance_decomposition'), C.variance_decomposition = o.variance_decomposition; end
    if isfield(o, 'conditional_variance_decomposition')
        C.conditional_variance_decomposition = o.conditional_variance_decomposition;
        C.cvd_steps = op.conditional_variance_decomposition;
    end
    if op.periods > 0
        C.endo_simul = o.endo_simul;   % endo_nbr x periods
        C.exo_simul = o.exo_simul;     % periods x exo_nbr
    end
    cmds{end+1} = C;
end
R.cmds = cmds;
fid = fopen('ref_nk.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('wrote ref_nk.json');
