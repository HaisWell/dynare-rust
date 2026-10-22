% dump_q2.m -- references: run Dynare on rbc_q2.mod for every CASE and
% write each command's oo_ to ref_q2_c<k>.json (logs kept as q2_c<k>_matlab.log).
% Run from a PRIVATE copy of reference/q2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q2.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end  % DYNARE_VERSION=7.1 dumps with Dynare 7.1
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('cases', 'var')
    cases = [1 2 3 4 5 6 7 8];
end
for cs = cases
    eval(sprintf('dynare rbc_q2 -DMATLAB_DUMP -DCASE=%d nointeractive noclearall', cs));
    copyfile('rbc_q2.log', sprintf('q2_c%d_matlab.log', cs));
    if exist(fullfile('rbc_q2', 'latex'), 'dir')
        copyfile(fullfile('rbc_q2', 'latex'), sprintf('latex_c%d', cs));
        rmdir(fullfile('rbc_q2', 'latex'), 's');
    end
    R = struct();
    R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.Sigma_e = M_.Sigma_e;
    R.steady_state = oo_.steady_state;
    cmds = {};
    k = 1;
    while exist(sprintf('q2_c%d_cmd%d.mat', cs, k), 'file')
        S = load(sprintf('q2_c%d_cmd%d.mat', cs, k));
        o = S.oo_; op = S.options_;
        C = struct();
        C.var_list = o.var_list;
        C.ghx = o.dr.ghx; C.ghu = o.dr.ghu; C.order_var = o.dr.order_var; C.ys = o.dr.ys;
        if isfield(o.dr, 'ghxx'), C.ghxx = o.dr.ghxx; C.ghuu = o.dr.ghuu; C.ghxu = o.dr.ghxu; C.ghs2 = o.dr.ghs2; end
        if isfield(o.dr, 'eigval'), C.eigval_re = real(o.dr.eigval); C.eigval_im = imag(o.dr.eigval); end
        C.periods = op.periods; C.drop = op.drop; C.ar = op.ar;
        fl = {'irfs','gamma_y','mean','var','autocorr','skewness','kurtosis','variance_decomposition', ...
              'conditional_variance_decomposition','contemporaneous_correlation','SpectralDensity'};
        for f = fl
            if isfield(o, f{1}), C.(f{1}) = o.(f{1}); end
        end
        if op.periods > 0
            C.endo_simul = o.endo_simul;
            C.exo_simul = o.exo_simul;
        end
        cmds{end+1} = C;
        delete(sprintf('q2_c%d_cmd%d.mat', cs, k));
        k = k + 1;
    end
    R.cmds = cmds;
    fid = fopen(sprintf('ref_q2_c%d.json', cs), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
    fprintf('wrote ref_q2_c%d.json\n', cs);
    clear oo_ M_ options_ oo_recursive_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info
end
