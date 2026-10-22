% Reference for tests/realtime_epilogue.rs: run from a private copy of this directory with
%   matlab -batch "run('dump.m')"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
S = struct();
try
    dynare rt_epi nointeractive nograph noclearall console
    S.ok = true;
    S.epilogue_names = M_.epilogue_names;
    S.endo_names = M_.endo_names;
    S.realtime_shock_decomposition = oo_.realtime_shock_decomposition;
    S.conditional_shock_decomposition = oo_.conditional_shock_decomposition;
    S.realtime_forecast_shock_decomposition = oo_.realtime_forecast_shock_decomposition;
    S.realtime_conditional_shock_decomposition = oo_.realtime_conditional_shock_decomposition;
    S.epilogue_steady_state = oo_.shock_decomposition_info.epilogue_steady_state;
catch err
    S.ok = false;
    S.error = err.message;
    S.stack = err.stack(1:min(5,numel(err.stack)));
end
fid = fopen('ref_rt_epi.json', 'w'); fprintf(fid, '%s', jsonencode(S, 'ConvertInfAndNaN', true)); fclose(fid);
exit(0);
