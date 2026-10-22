% make_datafiles.m -- writes the SW07 data (../sw07/sw07_obs.csv) as an .xlsx file and as a
% legacy .mat datafile, then reads them back the way Dynare does (dseries with sheet and
% range; load_mat_file_data_legacy) and dumps what Dynare sees to ref_datafiles.json.
% Run from a PRIVATE COPY of reference/ (it needs ../sw07/sw07_obs.csv):
%   cd data; /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('make_datafiles.m')"
% Dynare version: DYNARE_VERSION environment variable (default 7.2), e.g. DYNARE_VERSION=7.1 matlab -batch ...
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath([droot '/matlab']);
cd(fileparts(mfilename('fullpath')));
dynare_config();
T = readtable('../sw07/sw07_obs.csv', 'VariableNamingRule', 'preserve');
names = T.Properties.VariableNames(2:end);
X = table2array(T(:, 2:end));
n = size(X, 1);
q = arrayfun(@(k) sprintf('%dQ%d', 2002 + floor((k-1)/4), mod(k-1, 4) + 1), (1:n)', 'UniformOutput', false);
% sheet 'notes' first, data on sheet 'data' from B3 (rows 1-2 and column A hold junk/blank)
writecell({'this sheet is not the data'}, 'sw07_obs.xlsx', 'Sheet', 'notes');
C = [{''}, names; q, num2cell(X)];
writecell(C, 'sw07_obs.xlsx', 'Sheet', 'data', 'Range', 'B3');
writecell({'title row, outside the range'}, 'sw07_obs.xlsx', 'Sheet', 'data', 'Range', 'A1');
% legacy .mat datafile: one vector per variable
S = struct();
for j = 1:numel(names), S.(names{j}) = X(:, j); end
save('sw07_obs.mat', '-struct', 'S');

last = sprintf('%s%d', char('A' + numel(names) + 1), n + 3);   % column B + numel(names)
% what makedataset -> dseries -> load_data does with options_.xls_sheet / xls_range
[~, init, xdata, xnames] = load_xls_file_data('sw07_obs.xlsx', 2, ['B3:' last]);  % xlsread without Excel (macOS) takes sheet numbers only
dm = load_mat_file_data_legacy('sw07_obs.mat', names);
R = struct('xls_range', ['B3:' last], 'names', {xnames}, 'first_date', char(init), 'data', xdata, ...
           'mat_names', {dm.name}, 'mat_data', dm.data);
fid = fopen('ref_datafiles.json', 'w'); fprintf(fid, '%s', jsonencode(R, 'ConvertInfAndNaN', true)); fclose(fid);
disp('datafiles written');
