% dump_ss2.m -- stoch_simul completion references: run Dynare 7.2 on pi_nk.mod and rbc_ll.mod
% for every CASE and write each command's oo_ to ref_<model>_c<k>.json (logs kept as
% <model>_c<k>_matlab.log, errors in the json's `error` field).
% Run from a PRIVATE copy of reference/stoch_simul2:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_ss2.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
if ~exist('runs', 'var')
    runs = {{'pi_nk', 1:11}, {'rbc_ll', 1:9}};
end
for r = 1:numel(runs)
    model = runs{r}{1};
    for cs = runs{r}{2}
        errmsg = '';
        try
            eval(sprintf('dynare %s -DMATLAB_DUMP -DCASE=%d nointeractive noclearall', model, cs));
        catch e
            errmsg = e.message;
            fprintf('CASE %d error: %s\n', cs, errmsg);
            diary off
        end
        copyfile([model '.log'], sprintf('%s_c%d_matlab.log', model, cs));
        if exist(fullfile(model, 'graphs', [model '_IRF.tex']), 'file')
            movefile(fullfile(model, 'graphs', [model '_IRF.tex']), sprintf('%s_c%d_IRF.tex', model, cs));
        end
        R = struct();
        R.error = errmsg;
        R.endo_names = M_.endo_names; R.exo_names = M_.exo_names; R.Sigma_e = M_.Sigma_e;
        cmds = {};
        k = 1;
        while exist(sprintf('ss2_c%d_cmd%d.mat', cs, k), 'file')
            S = load(sprintf('ss2_c%d_cmd%d.mat', cs, k));
            o = S.oo_; op = S.options_;
            C = struct();
            C.var_list = o.var_list;
            C.ghx = o.dr.ghx; C.ghu = o.dr.ghu; C.ys = o.dr.ys;
            if isfield(o.dr, 'order_var'), C.order_var = o.dr.order_var; end
            if isfield(o.dr, 'eigval'), C.eigval_re = real(o.dr.eigval); C.eigval_im = imag(o.dr.eigval); end
            pif = {'PI_ghx','PI_ghu','PI_TT1','PI_TT2','PI_nmat','PI_FL_RANK'};
            for f = pif
                if isfield(o.dr, f{1}), C.(f{1}) = o.dr.(f{1}); end
            end
            C.partial_information = op.partial_information; C.aim_solver = op.aim_solver;
            fl = {'irfs','gamma_y','mean','var','autocorr','variance_decomposition', ...
                  'conditional_variance_decomposition','contemporaneous_correlation', ...
                  'variance_decomposition_ME','conditional_variance_decomposition_ME'};
            for f = fl
                if isfield(o, f{1}), C.(f{1}) = o.(f{1}); end
            end
            cmds{end+1} = C;
            delete(sprintf('ss2_c%d_cmd%d.mat', cs, k));
            k = k + 1;
        end
        R.cmds = cmds;
        if exist(fullfile(model, 'Output', [model '_PCL_moments.mat']), 'file')
            P = load(fullfile(model, 'Output', [model '_PCL_moments.mat']));
            R.PCL_moments = P;
            delete(fullfile(model, 'Output', [model '_PCL_moments.mat']));
        end
        fid = fopen(sprintf('ref_%s_c%d.json', model, cs), 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
        fprintf('wrote ref_%s_c%d.json\n', model, cs);
        clear oo_ M_ options_ oo_recursive_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info
    end
end
