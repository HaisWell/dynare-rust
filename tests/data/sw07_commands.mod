// Stream F commands after the SW07 STAGE 9 estimation (mode file, smoother, forecast=12).
// Used by tests/forecast_commands.rs, which copies it next to reference/sw07/sw07.mod in a temp dir.
@#define STAGE = 9
@#include "sw07.mod"

shock_groups(name=main);
'Prices' = epinf, ew;
demand = eb, eg, eqs;
end;

shock_decomposition(use_shock_groups=main, nograph) pinf4 y;
forecast(periods=8, conf_sig=0.8, nograph) pinf4 y;
calib_smoother(datafile='sw07_obs.csv', first_obs=5, presample=8, diffuse_filter, filtered_vars) pinf4;
