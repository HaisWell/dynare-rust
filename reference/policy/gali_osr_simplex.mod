// gali_osr.mod with MATLAB's Nelder-Mead simplex (opt_algo=7) instead of csminwel, optimizing
// phi_y only (with both coefficients free the loss keeps falling as they grow without bound
// in a fixed ratio, and the simplex just drifts along that ridge).
@#define OPT_ALGO = 7
@#define ONLY_PHI_Y = 1
@#include "gali_osr.mod"
