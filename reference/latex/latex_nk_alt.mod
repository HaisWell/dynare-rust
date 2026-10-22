// Alternative to latex_nk.mod for model_comparison: phi_pi calibrated, tighter kappa prior.
@#include "nk_model.inc"
varobs y pi i;

estimated_params;
rho_y, beta_pdf, 0.6, 0.15;
kappa, gamma_pdf, 0.3, 0.1;
rho_i, beta_pdf, 0.5, 0.2;
stderr e_y, inv_gamma_pdf, 0.8, inf;
stderr e_pi, inv_gamma_pdf, 0.5, inf;
stderr e_i, inv_gamma_pdf, 0.3, inf;
end;

estimation(datafile = nk_data, mode_compute = 4, mh_replic = 0, nograph);
