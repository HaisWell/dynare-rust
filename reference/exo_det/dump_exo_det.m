% dump_exo_det.m -- varexo_det references (Dynare 7.2 by default, DYNARE_VERSION): for every file of names.txt, the
% console output and error message (<name>.log) and ref_<name>.json (decision rules with
% ghud/ghxud/ghuud/ghudud, oo_.exo_det_simul, forecasts, smoother, ...). For the order-2
% forecasts, the forecast is re-run with options_.forecast_replic = 50 after rng(7) and the
% normals it draws are saved (replay). The preprocessor's error messages (<name>.pp.txt) and
% the JSON of xdet_json.mod (json/) come from dump_pp.sh. Run from a PRIVATE copy of
% reference/exo_det:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_exo_det.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
here = fileparts(mfilename('fullpath'));
cd(here);
names = strsplit(strtrim(fileread('names.txt')));
for q = 1:numel(names)
    nm = names{q};
    clear global; global M_ options_ oo_ %#ok<TLEV>
    rng('default');
    txt = evalc(['try, dynare ' nm ' nointeractive nograph noclearall, catch ME__, disp([''ERRMSG: '' ME__.message]), end']);
    err = '';
    i = strfind(txt, 'ERRMSG: ');
    if ~isempty(i), err = strtrim(txt(i(end)+8:end)); end
    fid = fopen([nm '.log'], 'w'); fprintf(fid, '%s', txt); fclose(fid);
    R = struct();
    R.err = err;
    try, R.exo_names = M_.exo_names; R.exo_det_names = M_.exo_det_names; R.param_names = M_.param_names; end %#ok<*TRYNC>
    try, R.ys = oo_.dr.ys; R.order_var = oo_.dr.order_var; end
    f = {'ghx','ghu','ghxx','ghxu','ghuu','ghs2','ghud','ghxud','ghuud','ghudud','exo_det_length'};
    for i = 1:numel(f), try, R.(f{i}) = oo_.dr.(f{i}); end, end
    g = {'exo_det_simul','exo_det_steady_state','exo_simul','endo_simul','forecast','irfs','steady_state', ...
         'SmoothedVariables','SmoothedShocks','shock_decomposition','conditional_forecast','MarginalDensity'};
    for i = 1:numel(g), try, R.(g{i}) = oo_.(g{i}); end, end
    try, R.Sigma_e = M_.Sigma_e; R.params = M_.params; end
    if any(strcmp(nm, {'xdet_o2', 'xdet_o2m', 'xdet_nox_o2', 'xdet_nox_o3'})) && isempty(err)
        % replay: the order-2 forecast bands with 50 simulations on known normals
        opts = options_; opts.forecast_replic = 50;
        rng(7);
        R.replay_forecast = forecasts.run(var_list_, M_, opts, oo_, 'simul');
        rng(7);
        R.replay_normals = randn(1, 50*M_.exo_nbr*40);
        R.replay_periods = options_.periods;
    end
    if strcmp(nm, 'xdet_latex')
        if ~isfolder(fullfile(here, 'latex')), mkdir(fullfile(here, 'latex')); end
        copyfile('xdet_latex/latex/xdet_latex_latex_definitions.tex', 'latex/xdet_latex_latex_definitions.tex');
        copyfile('xdet_latex/latex/xdet_latex_latex_parameters.tex', 'latex/xdet_latex_latex_parameters.tex');
        copyfile('xdet_latex/latex/dynamic_content.tex', 'latex/dynamic_content.tex');
    end
    if strcmp(nm, 'xdet_est') && isempty(err)
        try, R.mode = oo_.posterior_mode; end
    end
    fid = fopen(['ref_' nm '.json'], 'w');
    fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true));
    fclose(fid);
end
% the preprocessor's own messages and JSON: dump_pp.sh
