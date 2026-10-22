// New Keynesian model with a zero lower bound on the nominal interest rate (one OccBin
// constraint). The policy rule is smoothed on the actual rate, so the regime changes the
// state dynamics. The shadow (notional) rate `inot` is used to leave the ZLB regime.
// Variants (macro defines) exercise the solver options; see dump_occbin.m.

var y pi i inot r_nat u_s;
varexo eps_d eps_s;
parameters beta sigma kappa phi_pi phi_y rho_i rho_d rho_s istar ilb;

beta   = 0.99;
sigma  = 1;
kappa  = 0.1;
phi_pi = 1.5;
phi_y  = 0.125;
rho_i  = 0.6;
rho_d  = 0.85;
rho_s  = 0.5;
istar  = 1/beta - 1;
ilb    = 0;

model;
[name='IS']
y = y(+1) - 1/sigma*(i - pi(+1) - istar - r_nat);
[name='PC']
pi = beta*pi(+1) + kappa*y + u_s;
r_nat = rho_d*r_nat(-1) + eps_d;
u_s = rho_s*u_s(-1) + eps_s;
inot = rho_i*i(-1) + (1-rho_i)*(istar + phi_pi*pi + phi_y*y);
[name='mp', relax='zlb']
i = inot;
[name='mp', bind='zlb']
i = ilb;
end;

occbin_constraints;
name 'zlb'; bind inot < ilb; relax inot > ilb;
end;

initval;
y = 0; pi = 0; i = istar; inot = istar; r_nat = 0; u_s = 0;
end;
steady;

shocks;
var eps_d; stderr 0.01;
var eps_s; stderr 0.002;
end;

shocks(surprise);
var eps_d;
periods 1, 2:3, 15, 30;
values -0.03, -0.004, -0.025, 0.01;
var eps_s;
periods 2, 20:21;
values 0.002, -0.003;
end;

occbin_setup;
@#ifndef VARIANT
occbin_solver(simul_periods=80, simul_check_ahead_periods=40);
@#else
@#if VARIANT == 1
occbin_solver(simul_periods=60, simul_check_ahead_periods=20, simul_curb_retrench, simul_maxit=50);
@#elseif VARIANT == 2
// check_ahead shorter than the ZLB spell: nperiods is increased endogenously
occbin_solver(simul_periods=40, simul_check_ahead_periods=3);
@#elseif VARIANT == 3
occbin_solver(simul_periods=50, simul_check_ahead_periods=8, simul_reset_regime_in_new_period, simul_max_check_ahead_periods=11);
@#elseif VARIANT == 4
// myopic agents: regimes are checked only 5 periods ahead (shorter than the ZLB spell)
occbin_solver(simul_periods=50, simul_check_ahead_periods=4, simul_max_check_ahead_periods=5);
@#elseif VARIANT == 5
// too few iterations: error 311
occbin_solver(simul_periods=30, simul_maxit=2);
@#endif
@#endif
occbin_write_regimes;
occbin_graph y pi i inot eps_d;
