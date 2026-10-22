function dump_stoch(name)
% Runs <name>.mod with Dynare 7.2 and writes ref_<name>.json
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
evalin('base', ['dynare ' name ' noclearall nograph']);
M_ = evalin('base','M_'); oo_ = evalin('base','oo_'); options_ = evalin('base','options_');
r = struct();
r.endo_names = M_.endo_names;
r.exo_names = M_.exo_names;
r.params = M_.params;
r.steady_state = oo_.steady_state;
r.ys = oo_.dr.ys; r.order_var = oo_.dr.order_var;
r.ghx = oo_.dr.ghx; r.ghu = oo_.dr.ghu;
for f = {'ghxx','ghxu','ghuu','ghs2'}
  if isfield(oo_.dr, f{1}), r.(f{1}) = oo_.dr.(f{1}); end
end
if options_.periods > 0, r.exo_simul = oo_.exo_simul; r.drop = options_.drop; end
if isfield(oo_,'mean'), r.mean = oo_.mean; end
if isfield(oo_,'var'), r.var = oo_.var; end
if isfield(oo_,'autocorr'), r.autocorr = oo_.autocorr; end
if isfield(oo_,'variance_decomposition'), r.variance_decomposition = oo_.variance_decomposition; end
if isfield(oo_,'irfs'), r.irfs = oo_.irfs; end
if isfield(oo_,'endo_simul'), r.endo_simul_size = size(oo_.endo_simul); end

fid = fopen(['ref_' name '.json'],'w'); fprintf(fid,'%s',jsonencode(r)); fclose(fid);
end
