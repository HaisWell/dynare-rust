% dump_occbin.m -- reference values for tests/occbin.rs, written to ref_occbin.json.
% Runs the OccBin models of this folder (and variants via -DVARIANT=k) and stores
% oo_.occbin.simul (piecewise, linear, regime history, shocks) plus the state-space
% matrices T, R, C of occbin.solver's third output and the dynamic Jacobians of each regime.
% Run from a private copy of this folder (Dynare writes +fname/ and fname/ next to the .mod):
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch "dump_occbin"
% DYNARE_VERSION=7.1 (environment) dumps with Dynare 7.1; the default is 7.2
ver = getenv('DYNARE_VERSION'); if isempty(ver), ver = '7.2'; end
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', ver); end;
addpath(fullfile(droot, 'matlab'));
here = fileparts(mfilename('fullpath'));
cd(here);
runs = { 'nk_zlb', '', 'nk_zlb';
         'nk_zlb', '-DVARIANT=1', 'nk_zlb_v1';
         'nk_zlb', '-DVARIANT=2', 'nk_zlb_v2';
         'nk_zlb', '-DVARIANT=3', 'nk_zlb_v3';
         'nk_zlb', '-DVARIANT=4', 'nk_zlb_v4';
         'nk_zlb', '-DVARIANT=5', 'nk_zlb_v5';
         'borrcon', '', 'borrcon';
         'rbc_2c', '', 'rbc_2c';
         'rbc_2c', '-DVARIANT=1', 'rbc_2c_v1' };
out = struct();
global M_ oo_ options_
for k = 1:size(runs,1)
    clear occbin.solver occbin.solve_one_constraint occbin.solve_two_constraints ...
        occbin.mkdatap_anticipated_dyn occbin.mkdatap_anticipated_2constraints_dyn
    clear functions
    try
        if isempty(runs{k,2})
            dynare(runs{k,1}, 'noclearall', 'nograph', 'nolog');
        else
            dynare(runs{k,1}, 'noclearall', 'nograph', 'nolog', runs{k,2});
        end
    catch err
        % OccBin errors go through print_info -> error(); keep the message
        out.(runs{k,3}) = struct('error', err.message);
        continue
    end

    s = oo_.occbin.simul;
    r = struct();
    r.piecewise = s.piecewise;
    r.linear = s.linear;
    r.ys = s.ys;
    r.shocks_sequence = s.shocks_sequence;
    r.exo_pos = s.exo_pos;
    r.endo_names = M_.endo_names;
    r.param_names = M_.param_names;
    r.params = M_.params;
    r.pswitch = M_.occbin.pswitch;
    rh = s.regime_history;
    if M_.occbin.constraint_nbr == 1
        r.regime = arrayfun(@(x) double(x.regime), rh, 'UniformOutput', false);
        r.regimestart = arrayfun(@(x) double(x.regimestart), rh, 'UniformOutput', false);
    else
        r.regime1 = arrayfun(@(x) double(x.regime1), rh, 'UniformOutput', false);
        r.regimestart1 = arrayfun(@(x) double(x.regimestart1), rh, 'UniformOutput', false);
        r.regime2 = arrayfun(@(x) double(x.regime2), rh, 'UniformOutput', false);
        r.regimestart2 = arrayfun(@(x) double(x.regimestart2), rh, 'UniformOutput', false);
    end
    % the state space of each shock period (same persistent state as the run above)
    [~, ~, ss] = occbin.solver(M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    r.T = ss.T; r.R = ss.R; r.C = ss.C;
    [h_m1, h0, h_p1, h_exo] = occbin.get_deriv(M_, oo_.steady_state);
    r.g1 = [h_m1 h0 h_p1 h_exo];
    Ms = M_; Ms.params(M_.occbin.pswitch) = 1;
    [h_m1, h0, h_p1, h_exo, res] = occbin.get_deriv(Ms, oo_.steady_state);
    r.g1_star = [h_m1 h0 h_p1 h_exo];
    r.resid_star = res;
    out.(runs{k,3}) = r;
end
fid = fopen(fullfile(here, 'ref_occbin.json'), 'w');
fprintf(fid, '%s', jsonencode(out, 'ConvertInfAndNaN', true));
fclose(fid);
disp('wrote ref_occbin.json');
