% Reference for smoother2histval with the default invars (AUX_ENDO_LAG and AUX_EXO_LAG rows of
% M_.endo_histval) and the forecast that starts from it, after an estimation (s2h.mod ->
% ref_s2h.json) and after calib_smoother (s2h_calib.mod -> ref_s2h_calib.json).
% Run from a private copy of reference/s2h:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('dump_s2h.m')"
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
for mod = {'s2h', 's2h_calib'}
clearvars -global
global M_ oo_ options_
dynare(mod{1}, 'noclearall', 'nolog', 'nograph', 'nointeractive')
r0 = fieldnames(oo_.SmoothedVariables);
r = struct();
r.endo_names = M_.endo_names;
r.smoothed_fields = r0;
r.endo_histval = M_.endo_histval;
r.exo_histval = M_.exo_histval;
r.forecast_y = oo_.forecast.Mean.y;
r.forecast_z = oo_.forecast.Mean.z;
r.smoothed_shock_u = oo_.SmoothedShocks.u;
fid = fopen(['ref_' mod{1} '.json'], 'w');
fprintf(fid, '%s', jsonencode(r, 'ConvertInfAndNaN', true));
fclose(fid);
end
