function r = est_dump(M_, oo_, options_, bayestopt_)
% Key estimation results for comparison with dynare-rs.
r = struct();
r.steady_state = oo_.steady_state;
if isfield(oo_,'posterior_mode')
  r.mode_params = oo_.posterior_mode.parameters;
  if isfield(oo_.posterior_mode,'shocks_std'), r.mode_shocks_std = oo_.posterior_mode.shocks_std; end
  r.mode_sd_params = oo_.posterior_std_at_mode.parameters;
  if isfield(oo_.posterior_std_at_mode,'shocks_std'), r.mode_sd_shocks_std = oo_.posterior_std_at_mode.shocks_std; end
end
if isfield(oo_,'MarginalDensity')
  r.MarginalDensity = oo_.MarginalDensity;
end
if isfield(oo_,'posterior') && isfield(oo_.posterior,'optimization')
  r.optimization = oo_.posterior.optimization;
end
if isfield(oo_,'likelihood_at_initial_parameters'), r.lik_init = oo_.likelihood_at_initial_parameters; end
if isfield(oo_,'posterior_mean'), r.posterior_mean = oo_.posterior_mean; r.posterior_std = oo_.posterior_std; end
r.xparam1_names = bayestopt_.name;
if isfield(oo_,'dr') && isfield(oo_.dr,'ghx'), r.ghx = oo_.dr.ghx; r.ghu = oo_.dr.ghu; r.order_var = oo_.dr.order_var; end
end
