// realtime_shock_decomposition(with_epilogue) reference (dump.m): epilogue with lags
// (ya = sum of exp(y) over 4 quarters, gya = growth of ya), vintages of the realtime recursion,
// conditional 1-step / k-step decompositions, realtime forecast and realtime conditional.
var y c k;
varexo e;
parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;
model;
c + k = exp(y)*k(-1)^alpha + (1-delta)*k(-1);
y = rho*y(-1) + e;
1/c = beta/c(+1)*(alpha*exp(y(+1))*k^(alpha-1)+1-delta);
end;
epilogue;
ya = exp(y)+exp(y(-1))+exp(y(-2))+exp(y(-3));
gya = ya/ya(-4)-1;
end;
steady_state_model;
y = 0; k = ((1/beta-1+delta)/alpha)^(1/(alpha-1)); c = k^alpha - delta*k;
end;
shocks; var e; stderr 0.01; end;
varobs y;
estimated_params;
rho, beta_pdf, 0.9, 0.05;
end;
// (a datafile= estimation: with the data command MATLAB's smoother ignores options_.nobs,
// so every vintage would be the full sample)
estimation(datafile='q4_data.csv', first_obs=6, nobs=31, mode_compute=0, smoother, nograph, plot_priors=0);
realtime_shock_decomposition(parameter_set=calibration, forecast=4, save_realtime=[6 10 14 20 24 28 31], with_epilogue) y;
plot_shock_decomposition(nograph, realtime=1, fig_name='rt1') y ya gya;
plot_shock_decomposition(nograph, realtime=2, vintage=20, fig_name='rt2') y ya gya;
plot_shock_decomposition(nograph, realtime=3, vintage=20, fig_name='rt3') y ya gya;
