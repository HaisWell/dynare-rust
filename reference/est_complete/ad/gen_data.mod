@#include "ad_model.inc"
varobs y c;
set_dynare_seed(7);
stoch_simul(order=1, periods=400, drop=200, irf=0, nograph, noprint);
