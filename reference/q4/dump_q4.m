% dump_q4.m -- reference for dynare-rust: dates, data command / set_time / ISO
% csv dates, epilogue and shock_decomposition(with_epilogue), plot_shock_decomposition
% (write_xls, steadystate, init2shocks), initval_file / histval_file with dates.
% Run from a PRIVATE copy of reference/q4:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_q4.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
dynare_config;

% ---- dates ----------------------------------------------------------------------------
S = struct();
strs = {'2004Q1','2004M3','2004M12','2004Y','2004A','2004S2','2004H1','2020W53','2004W1','2004-03-31','2003-03-31','1990Q4','2004q2'};
for i = 1:numel(strs)
    d = dates(strs{i});
    S.dates(i).input = strs{i};
    S.dates(i).label = char(d);
    S.dates(i).freq = d.freq;
    S.dates(i).time = d.time;
    S.dates(i).plus5 = char(d+5);
    S.dates(i).minus3 = char(d-3);
end
w = dates('2015W50'):dates('2016W3');
S.weekly_range = cellfun(@(k) char(w(k)), num2cell(1:length(w)), 'UniformOutput', false);
S.isdate = cellfun(@isdate, {'2004Q5','2004M13','2004-3-31','2004-02-30','2003W53','2004','2004QQ1','2004-02-29'});
fid = fopen('ref_q4_dates.json', 'w'); fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true)); fclose(fid);

% ---- estimation cases ------------------------------------------------------------------
global M_ options_ oo_ dataset_
for c = 1:4
    S = struct();
    try
        eval(sprintf('dynare q4_est -DCASE=%d nointeractive nograph noclearall console', c));
        S.ok = true;
        S.dataset_init = char(dataset_.dates(1));
        S.dataset_last = char(dataset_.dates(end));
        S.nobs = dataset_.nobs;
        S.smoothed_y = oo_.SmoothedVariables.y;
        S.endo_names = M_.endo_names;
        S.epilogue_names = M_.epilogue_names;
        S.shock_decomposition = oo_.shock_decomposition;
        if isfield(oo_.shock_decomposition_info, 'epilogue_steady_state')
            S.epilogue_steady_state = oo_.shock_decomposition_info.epilogue_steady_state;
        end
        if isfield(oo_, 'initval_decomposition')
            S.initval_decomposition = oo_.initval_decomposition;
        end
        if c == 1 || c == 3
            % the epilogue over the smoothed variables (dynare-rust writes it after the smoother)
            ds = dseries([oo_.SmoothedVariables.y oo_.SmoothedVariables.c oo_.SmoothedVariables.k], '1991Q2', {'y'; 'c'; 'k'});
            ds = q4_est.epilogue_dynamic(M_.params, ds);
            for i = 1:numel(M_.epilogue_names)
                S.epilogue_smoothed.(M_.epilogue_names{i}) = ds.(M_.epilogue_names{i}).data;
            end
        end
        f = dir(fullfile('q4_est', 'Output', 'q4_est_shock_decomposition*'));
        for i = 1:numel(f)
            fn = fullfile(f(i).folder, f(i).name);
            S.xls(i).name = f(i).name;
            sh = sheetnames(fn);
            for j = 1:numel(sh)
                S.xls(i).sheets(j).name = sh{j};
                C = readcell(fn, 'Sheet', sh{j});
                C(cellfun(@(x) isa(x, 'missing'), C)) = {[]};
                S.xls(i).sheets(j).cells = C;
            end
            delete(fn);
        end
    catch err
        S.ok = false;
        S.error = err.message;
    end
    fid = fopen(sprintf('ref_q4_est_case%d.json', c), 'w'); fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true)); fclose(fid);
    fprintf('case %d done\n', c);
end

% ---- initval_file / histval_file with dates -------------------------------------------
for c = 1:3
    S = struct();
    try
        eval(sprintf('dynare q4_datafiles -DCASE=%d nointeractive nograph noclearall console', c));
        S.ok = true;
        S.endo_simul = oo_.endo_simul;
        S.exo_simul = oo_.exo_simul;
        S.periods = options_.periods;
        S.endo_histval = M_.endo_histval;
    catch err
        S.ok = false;
        S.error = err.message;
    end
    fid = fopen(sprintf('ref_q4_datafiles_case%d.json', c), 'w'); fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true)); fclose(fid);
    fprintf('datafiles case %d done\n', c);
end
