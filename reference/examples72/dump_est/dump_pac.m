droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
base = getenv('E72_WORK'); if isempty(base), base = fullfile(tempdir, 'e72', 'm'); end
% pac_head holds pac_model.mod cut after pac.update.expectation (line 72)
cd(fullfile(base,'pac_head'));
dynare pac_model nograph nointeractive
r = struct(); r.param_names = M_.param_names; r.params = M_.params;
fid=fopen(fullfile(base,'pac_model.json'),'w'); fprintf(fid,'%s',jsonencode(r,'ConvertInfAndNaN',true)); fclose(fid);
% the whole example (pac_full holds the shipped pac_model.mod); writes pac_model_shocks.csv
cd(fullfile(base,'pac_full'));
dynare pac_model nograph nointeractive
% the innovations drawn by simul_backward_model (MATLAB's RNG), for the port to replay
shocks = TrueData{M_.exo_names{:}}(TrueData.dates(11):TrueData.dates(end));
fid = fopen('pac_model_shocks.csv', 'w'); fprintf(fid, ',%s', strjoin(shocks.name', ',')); fprintf(fid, '\n');
for t=1:shocks.nobs, fprintf(fid, '%s', char(shocks.dates(t))); fprintf(fid, ',%.17g', shocks.data(t,:)); fprintf(fid, '\n'); end
fclose(fid);
% replaying them reproduces the simulation
p_est = M_.params;
pre = jsondecode(fileread(fullfile(base,'pac_model.json'))); M_.params = pre.params; % calibration (pac.update.expectation done)
sim2 = simul_backward_model(initialconditions, 5000, dseries('pac_model_shocks.csv'));
r = struct();
r.replay_maxdiff = max(max(abs(sim2.data - TrueData.data)));
r.endo_names = M_.endo_names; r.orig_endo_nbr = M_.orig_endo_nbr;
r.aux_types = [M_.aux_vars.type]; r.aux_exprs = {M_.aux_vars.orig_expr};
r.sim_names = TrueData.name; r.sim_init = char(TrueData.init);
r.sim_rows = [1 2 10 11 12 100 1000 2500 4999 5000 5009 5010];
r.sim_data = TrueData.data(r.sim_rows, :);
r.sim_colsum = sum(TrueData.data(11:end,:), 1);
r.sim_every50 = TrueData.data(50:50:end,:);
p = oo_.pac.pacman;
r.parnames = p.parnames; r.estimator = p.estimator; r.ssr = p.ssr; r.s2 = p.s2; r.R2 = p.R2;
r.covariance = p.covariance; r.student = p.student;
r.param_names = M_.param_names; r.params = p_est;
fid=fopen(fullfile(base,'pac_model_full.json'),'w'); fprintf(fid,'%s',jsonencode(r,'ConvertInfAndNaN',true)); fclose(fid);
