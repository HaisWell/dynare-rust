droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
base = getenv('E72_WORK'); if isempty(base), base = fullfile(tempdir, 'e72', 'm'); end
names = {'perfect_foresight_rbc','perfect_foresight_expectation_errors','rbc_occbin'};
for i=1:numel(names)
  n = names{i}; cd(fullfile(base,n));
  try
    eval(['dynare ' n ' nograph nointeractive']);
    r = struct();
    r.steady_state = oo_.steady_state; r.endo_names = M_.endo_names;
    if isfield(oo_,'endo_simul'), r.endo_simul = oo_.endo_simul; r.exo_simul = oo_.exo_simul; end
    if isfield(oo_,'occbin') && isfield(oo_.occbin,'simul')
      r.occbin_piecewise = oo_.occbin.simul.piecewise; r.occbin_linear = oo_.occbin.simul.linear;
      r.occbin_regime = oo_.occbin.simul.regime_history;
    end
    if isfield(oo_,'dr') && isfield(oo_.dr,'ghx'), r.ghx=oo_.dr.ghx; r.ghu=oo_.dr.ghu; r.order_var=oo_.dr.order_var; end
    fid=fopen(fullfile(base,[n '.json']),'w'); fprintf(fid,'%s',jsonencode(r,'ConvertInfAndNaN',true)); fclose(fid);
  catch e
    disp(getReport(e));
  end
  clearvars -except base names i
end
