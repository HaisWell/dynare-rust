% seeds_irfm.m -- rbc_irf_matching part 2 (Bayesian IRF matching, slice sampler, 2 x 50 draws)
% under several seeds, to show that which labelling of the two AR roots the chains settle on
% is decided by the random draws (tests/examples72_est.rs, rbc_irf_matching_bayesian_slice).
%
% Run from a scratch copy holding rbc_irf_matching_data.csv and p12s.mod, i.e.
% rbc_irf_matching_part12.mod with plot_priors = 0 and, just before the second
% method_of_moments call:
%     @#ifdef SEED
%     set_dynare_seed(@{SEED});
%     @#endif
% Writes "seed root_g_1 root_g_2 MHM" lines to seeds.txt (seed 0 = Dynare's default stream).
droot = getenv('DYNARE_ROOT'); if isempty(droot), droot = sprintf('/Applications/Dynare/%s-arm64', '7.2'); end;
addpath(fullfile(droot, 'matlab'));
seeds = 0:63;
fid = fopen('seeds.txt', 'w');
for s = seeds
    if s == 0, eval('dynare p12s nograph nointeractive'); else, eval(sprintf('dynare p12s nograph nointeractive -DSEED=%d', s)); end
    pm = oo_.mom.posterior_mean.parameters; md = oo_.mom.MarginalDensity.ModifiedHarmonicMean;
    fprintf(fid, '%d %.6f %.6f %.6f\n', s, pm.root_g_1, pm.root_g_2, md);
    clearvars -except seeds fid s
end
fclose(fid);
