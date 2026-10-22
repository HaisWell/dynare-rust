// Dynare (MATLAB) twin of nk_mcmc.mod, copied as nk_mcmc.mod into the MATLAB folders of
// dump_mcmc.m: same model, same file size (options_.MaxNumberOfBytes = 20000, i.e. 278 draws
// per mh file), Dynare's option names.
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

@#ifndef STAGE
@#define STAGE = 1
@#endif
@#ifndef REPLIC
@#define REPLIC = 3000
@#endif

options_.MaxNumberOfBytes = 20000;

@#if STAGE == 1
estimation(datafile = nk_data, mode_compute = 4, mh_replic = @{REPLIC}, mh_nblocks = 2, mh_jscale = 0.6,
           nograph, nodiagnostic);
@#elseif STAGE == 2
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', load_mh_file, mh_replic = @{REPLIC},
           mh_nblocks = 2, mh_jscale = 0.6, nograph, nodiagnostic);
@#elseif STAGE == 3
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', load_mh_file, mh_replic = 0,
           mh_nblocks = 2, nograph, nodiagnostic);
@#endif
