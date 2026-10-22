// MCMC persistence and continuation : Dynare's mh files, load_mh_file, mh_recover,
// mh_initialize_from_previous_mcmc, use_mh_covariance_matrix.
// -DSTAGE=1: mode + fresh MCMC; 2: continue it (load_mh_file); 3: post-process only
// (load_mh_file, mh_replic=0); 4: crash recovery (mh_recover); 5: continue with the previous
// posterior covariance (use_mh_covariance_matrix); 6: mh_posterior_mode_estimation;
// 7: mh_initialize_from_previous_mcmc from prev/nk_mcmc.
// max_number_of_bytes is dynare-rust's name for options_.MaxNumberOfBytes (small files).
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
@#ifndef BYTES
@#define BYTES = 20000
@#endif

@#if STAGE == 1
estimation(datafile = nk_data, mode_compute = 4, mh_replic = @{REPLIC}, mh_nblocks = 2, mh_jscale = 0.6,
           max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#elseif STAGE == 2
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', load_mh_file, mh_replic = @{REPLIC},
           mh_nblocks = 2, mh_jscale = 0.6, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#elseif STAGE == 3
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', load_mh_file, mh_replic = 0,
           mh_nblocks = 2, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#elseif STAGE == 4
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', mh_recover, mh_replic = @{REPLIC},
           mh_nblocks = 2, mh_jscale = 0.6, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#elseif STAGE == 5
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', load_mh_file, mh_replic = @{REPLIC},
           posterior_sampler_options = ('use_mh_covariance_matrix', 1),
           mh_nblocks = 2, mh_jscale = 0.6, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#elseif STAGE == 6
estimation(datafile = nk_data, mh_posterior_mode_estimation, mh_replic = @{REPLIC}, mh_nblocks = 1,
           prior_trunc = 1e-10, mh_jscale = 0.3, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#endif
@#if STAGE == 7
// a new MCMC that starts where the MCMC saved in prev/nk_mcmc stopped
estimation(datafile = nk_data, mode_compute = 0, mode_file = 'nk_mcmc/Output/nk_mcmc_mode', mh_replic = @{REPLIC},
           mh_initialize_from_previous_mcmc, mh_initialize_from_previous_mcmc_directory = 'prev/nk_mcmc',
           mh_nblocks = 2, mh_jscale = 0.6, max_number_of_bytes = @{BYTES}, nograph, nodiagnostic);
@#endif
@#if STAGE == 8
// Herbst-Schorfheide SMC (hssmc.m)
estimation(datafile = nk_data, posterior_sampling_method = 'hssmc',
           posterior_sampler_options = ('particles', @{REPLIC}, 'steps', 30), nograph);
@#endif
@#if STAGE == 9
// DIME ensemble MCMC (dime.m)
estimation(datafile = nk_data, posterior_sampling_method = 'dime_mcmc',
           posterior_sampler_options = ('nchain', 70, 'niter', @{REPLIC}, 'tune', 1000), nograph);
@#endif
