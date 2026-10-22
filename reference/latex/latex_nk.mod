// LaTeX output and model comparison reference model (stream B).
@#include "nk_model.inc"
varobs y pi i;

estimated_params;
rho_y, beta_pdf, 0.6, 0.15;
kappa, gamma_pdf, 0.15, 0.05;
phi_pi, 1.4, 1, 3, normal_pdf, 1.5, 0.25;
rho_i, beta_pdf, 0.5, 0.2;
stderr e_y, inv_gamma_pdf, 0.8, inf;
stderr e_pi, inv_gamma_pdf, 0.5, inf;
stderr e_i, uniform_pdf, , , 0, 2;
end;

write_latex_original_model;
write_latex_dynamic_model(write_equation_tags);
write_latex_static_model;
write_latex_definitions;
write_latex_parameter_table;
write_latex_prior_table;

estimation(datafile = nk_data, mode_compute = 4, mh_replic = 0, TeX, nograph);

model_comparison latex_nk_alt(0.4) latex_nk(0.6);
model_comparison(marginal_density = laplace) latex_nk latex_nk_alt;
collect_latex_files;
