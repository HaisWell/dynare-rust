// Simulates the data of fs2000_calib_smoother_cf.mod (see dump.m).
@#include "fs2000_model.inc"
stoch_simul(order=1, periods=300, drop=100, irf=0, nomoments, nocorr, nofunctions, noprint);
