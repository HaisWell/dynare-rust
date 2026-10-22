// Stream C scenario after the SW07 STAGE 9 estimation (mode file, smoother, forecast=12):
// a conditional forecast on the policy rate and long-run expected inflation over 8 quarters,
// realtime / initial-condition decompositions and plot_shock_decomposition variants.
// Used by tests/conditional_forecast_sw07.rs and reference/sw07/dump_conditional.m
// (copied next to sw07.mod; MATLAB runs the same file).
@#define STAGE = 9
@#include "sw07.mod"

shock_groups(name=main);
'Prices' = epinf, ew;
demand = eb, eg, eqs;
policy = em, epistar;
end;

// Scenario: the policy rate eased 1.6 -> 1.5 -> 1.4 -> 1.3, long-run expectations 0.9 then 0.85.
conditional_forecast_paths;
var robs;
periods 1:2, 3:4, 5:8;
values 1.5, 1.4, 1.3;
var pinflrobs;
periods 1:4 5:8;
values 0.9 (0.8+0.05);
end;

conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(em, e_lr),
                     periods=12, replic=5000, conf_sig=0.8);
plot_conditional_forecast(periods=12) robs pinflrobs pinf4 y dy;

realtime_shock_decomposition(forecast=4, save_realtime=[80 86 90]) pinf4 y r;
plot_shock_decomposition(realtime=1, vintage=86, type=yoy, fig_name=test, flip, use_shock_groups=main, nodisplay) pinf4;
plot_shock_decomposition(realtime=3, vintage=86, detail_plot, nodisplay) y;
initial_condition_decomposition(nodisplay) pinf4 r;
