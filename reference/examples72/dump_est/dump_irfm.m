droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
base = getenv('E72_WORK'); if isempty(base), base = fullfile(tempdir, 'e72', 'm'); end
addpath(base);
cd(fullfile(base,'rbc_irf_matching'));
copyfile('../../r/rbc_irf_matching/part1.mod','part1.mod');
copyfile('../../r/rbc_irf_matching/part12.mod','part12.mod');
names = {'part1','part12'};
for i=1:numel(names)
  try
    eval(['dynare ' names{i} ' nograph nointeractive']);
    r = struct(); r.mom = fullify(oo_.mom); r.params = M_.params; r.param_names = M_.param_names;
    if isfield(oo_,'posterior_mean'), r.posterior_mean = oo_.posterior_mean; end
    if isfield(oo_,'MarginalDensity'), r.MarginalDensity = oo_.MarginalDensity; end
    fid=fopen(fullfile(base,['irfm_' names{i} '.json']),'w'); fprintf(fid,'%s',jsonencode(r,'ConvertInfAndNaN',true)); fclose(fid);
  catch e
    disp(getReport(e));
  end
  clearvars -except base names i
end
