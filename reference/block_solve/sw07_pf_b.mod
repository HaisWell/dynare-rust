// Estimation stage switch (see ESTIMATION CALL at the end); no -D = 3, the MCMC.
@#ifndef STAGE
@#define STAGE = 3
@#endif
// =========================================================================
//  sw07.mod  --  Smets-Wouters (2007) with a drifting inflation target
// -------------------------------------------------------------------------
//  The estimation benchmark of dynare-rust (see docs/REFERENCES.md).
//  Written from the published log-linearised
//  equations of
//
//    Smets, F. and R. Wouters (2007), "Shocks and Frictions in US Business
//    Cycles: A Bayesian DSGE Approach", American Economic Review 97(3),
//    586-606 (model appendix and Table 1A/1B),
//
//  with the sticky-price economy, the flexible-price economy (potential
//  output) and the ARMA(1,1) price and wage mark-up shocks of the paper.
//  Parameter names follow the authors' replication file (c* prefix).
//
//  EXTENSIONS (unit roots, lead/lag auxiliaries, missing data, surveys):
//   1. A random-walk inflation target pistar (Del Negro, Giannoni and
//      Schorfheide 2015 use a near-unit-root target; here it is exactly a
//      unit root). The Phillips curve, the wage equation and the policy rule
//      are written in deviations from the target, so a permanent move in
//      pistar moves inflation and the nominal rate one for one and leaves the
//      real economy unchanged. Unit root => diffuse_filter.
//   2. A random-walk low-frequency component in hours (labtrend), seen only
//      through the hours observable: a second, measurement-side unit root.
//   3. The policy rule responds to four-quarter inflation pinf4 (lags 2-3 of
//      pinf: endogenous lag auxiliaries).
//   4. Two survey observables: one-year-ahead expected inflation (leads 2-4
//      of pinf: endogenous lead auxiliaries) and long-run expected inflation
//      (the target), each with its own measurement shock.
//   5. The data (sw07_obs.csv) are SIMULATED from this model at the
//      calibration below (gen_data.m, fixed seed); no real-world data. The two
//      survey series and dw are missing in the first quarters, which the
//      estimation skips with first_obs=5.
//  The steady state is found numerically from initval (no steady_state_model),
//  with steady(solve_algo=1).
// =========================================================================

var
    // flexible-price economy
    zcapf rkf kf pkf cf invef yf labf wf rrf kpf
    // sticky-price economy
    mc zcap rk k pk c inve y lab pinf w r kp
    // exogenous processes
    a b g qs ms spinf sw epinfma ewma
    // extensions
    pistar labtrend pinf4
    // observables
    dy dc dinve dw pinfobs robs labobs pinfexp4obs pinflrobs;

varexo ea eb eg eqs em epinf ew epistar elabt e_exp e_lr;

parameters
    ctou clandaw cg curvp curvw
    calfa csigma cfc cgy csadjcost chabb cprobw csigl cprobp cindw cindp czcap
    crpi crr cry crdy
    crhoa crhob crhog crhoqs crhoms crhopinf crhow cmap cmaw
    constepinf constebeta constelab ctrend;

// fixed in SW07
ctou    = 0.025;
clandaw = 1.5;
cg      = 0.18;
curvp   = 10;
curvw   = 10;

// SW07 posterior mode (Table 1A/1B), used as calibration and to simulate the data
calfa      = 0.19;
csigma     = 1.39;
cfc        = 1.61;
cgy        = 0.52;
csadjcost  = 5.48;
chabb      = 0.71;
cprobw     = 0.73;
csigl      = 1.92;
cprobp     = 0.65;
cindw      = 0.59;
cindp      = 0.22;
czcap      = 0.54;
crpi       = 2.03;
crr        = 0.81;
cry        = 0.08;
crdy       = 0.22;
crhoa      = 0.95;
crhob      = 0.18;
crhog      = 0.97;
crhoqs     = 0.71;
crhoms     = 0.12;
crhopinf   = 0.90;
crhow      = 0.97;
cmap       = 0.74;
cmaw       = 0.88;
constepinf = 0.81;
constebeta = 0.16;
constelab  = -0.1;
ctrend     = 0.43;


model(block);
// steady-state relations (SW07 appendix)
# cpie     = 1 + constepinf/100;
# cgamma   = 1 + ctrend/100;
# cbeta    = 1/(1 + constebeta/100);
# clandap  = cfc;
# cbetabar = cbeta*cgamma^(-csigma);
# cr       = cpie/(cbeta*cgamma^(-csigma));
# crk      = (cbeta^(-1))*(cgamma^csigma) - (1 - ctou);
# cw       = (calfa^calfa*(1 - calfa)^(1 - calfa)/(clandap*crk^calfa))^(1/(1 - calfa));
# cikbar   = 1 - (1 - ctou)/cgamma;
# cik      = (1 - (1 - ctou)/cgamma)*cgamma;
# clk      = ((1 - calfa)/calfa)*(crk/cw);
# cky      = cfc*clk^(calfa - 1);
# ciy      = cik*cky;
# ccy      = 1 - cg - cik*cky;
# crkky    = crk*cky;
# cwhlc    = (1/clandaw)*(1 - calfa)/calfa*crk*cky/ccy;
# conster  = (cr - 1)*100;
# chab     = chabb/cgamma;
# cpk      = 1/(cgamma^2*csadjcost);
# kappap   = ((1 - cprobp)*(1 - cbetabar*cgamma*cprobp)/cprobp)/((cfc - 1)*curvp + 1);
# kappaw   = (1 - cprobw)*(1 - cbetabar*cgamma*cprobw)/((1 + cbetabar*cgamma)*cprobw)/((clandaw - 1)*curvw + 1);

// =========================================================================
// FLEXIBLE-PRICE ECONOMY
// =========================================================================
a = calfa*rkf + (1 - calfa)*wf;
zcapf = (1/(czcap/(1 - czcap)))*rkf;
rkf = wf + labf - kf;
kf = kpf(-1) + zcapf;
invef = (1/(1 + cbetabar*cgamma))*(invef(-1) + cbetabar*cgamma*invef(+1) + cpk*pkf) + qs;
pkf = -rrf + (1/((1 - chab)/(csigma*(1 + chab))))*b
      + (crk/(crk + (1 - ctou)))*rkf(+1) + ((1 - ctou)/(crk + (1 - ctou)))*pkf(+1);
cf = chab/(1 + chab)*cf(-1) + (1/(1 + chab))*cf(+1)
     + ((csigma - 1)*cwhlc/(csigma*(1 + chab)))*(labf - labf(+1))
     - (1 - chab)/(csigma*(1 + chab))*rrf + b;
yf = ccy*cf + ciy*invef + g + crkky*zcapf;
yf = cfc*(calfa*kf + (1 - calfa)*labf + a);
wf = csigl*labf + (1/(1 - chab))*cf - (chab/(1 - chab))*cf(-1);
kpf = (1 - cikbar)*kpf(-1) + cikbar*invef + cikbar*cgamma^2*csadjcost*qs;

// =========================================================================
// STICKY-PRICE ECONOMY
// =========================================================================
mc = calfa*rk + (1 - calfa)*w - a;
zcap = (1/(czcap/(1 - czcap)))*rk;
rk = w + lab - k;
k = kp(-1) + zcap;
inve = (1/(1 + cbetabar*cgamma))*(inve(-1) + cbetabar*cgamma*inve(+1) + cpk*pk) + qs;
pk = -r + pinf(+1) + (1/((1 - chab)/(csigma*(1 + chab))))*b
     + (crk/(crk + (1 - ctou)))*rk(+1) + ((1 - ctou)/(crk + (1 - ctou)))*pk(+1);
c = chab/(1 + chab)*c(-1) + (1/(1 + chab))*c(+1)
    + ((csigma - 1)*cwhlc/(csigma*(1 + chab)))*(lab - lab(+1))
    - (1 - chab)/(csigma*(1 + chab))*(r - pinf(+1)) + b;
y = ccy*c + ciy*inve + g + crkky*zcap;
y = cfc*(calfa*k + (1 - calfa)*lab + a);
kp = (1 - cikbar)*kp(-1) + cikbar*inve + cikbar*cgamma^2*csadjcost*qs;

// Phillips curve and wage equation, inflation in deviations from the target
pinf - pistar = (1/(1 + cbetabar*cgamma*cindp))*(cbetabar*cgamma*(pinf(+1) - pistar)
                + cindp*(pinf(-1) - pistar(-1)) + kappap*mc) + spinf;
w = (1/(1 + cbetabar*cgamma))*w(-1) + (cbetabar*cgamma/(1 + cbetabar*cgamma))*w(+1)
    + (cindw/(1 + cbetabar*cgamma))*(pinf(-1) - pistar(-1))
    - (1 + cbetabar*cgamma*cindw)/(1 + cbetabar*cgamma)*(pinf - pistar)
    + (cbetabar*cgamma)/(1 + cbetabar*cgamma)*(pinf(+1) - pistar)
    + kappaw*(csigl*lab + (1/(1 - chab))*c - (chab/(1 - chab))*c(-1) - w) + sw;

// monetary policy: four-quarter inflation against the drifting target
pinf4 = (pinf + pinf(-1) + pinf(-2) + pinf(-3))/4;
r - pistar = crpi*(1 - crr)*(pinf4 - pistar) + cry*(1 - crr)*(y - yf)
             + crdy*(y - yf - y(-1) + yf(-1)) + crr*(r(-1) - pistar(-1)) + ms;

// =========================================================================
// EXOGENOUS PROCESSES
// =========================================================================
a = crhoa*a(-1) + ea;
b = crhob*b(-1) + eb;
g = crhog*g(-1) + eg + cgy*ea;
qs = crhoqs*qs(-1) + eqs;
ms = crhoms*ms(-1) + em;
spinf = crhopinf*spinf(-1) + epinfma - cmap*epinfma(-1);
epinfma = epinf;
sw = crhow*sw(-1) + ewma - cmaw*ewma(-1);
ewma = ew;
pistar = pistar(-1) + epistar;
labtrend = labtrend(-1) + elabt;

// =========================================================================
// MEASUREMENT EQUATIONS
// =========================================================================
dy = y - y(-1) + ctrend;
dc = c - c(-1) + ctrend;
dinve = inve - inve(-1) + ctrend;
dw = w - w(-1) + ctrend;
pinfobs = pinf + constepinf;
robs = r + conster;
labobs = lab + labtrend + constelab;
pinfexp4obs = constepinf + (pinf(+1) + pinf(+2) + pinf(+3) + pinf(+4))/4 + e_exp;
pinflrobs = constepinf + pistar + e_lr;
end;


// =========================================================================
// STEADY STATE
// =========================================================================
// Every model variable is a deviation from the balanced growth path, so its
// steady state is 0; the observables carry the SW07 constants. pistar and
// labtrend are unit roots: any level solves the static model, 0 is chosen.
initval;
dy          = ctrend;
dc          = ctrend;
dinve       = ctrend;
dw          = ctrend;
pinfobs     = constepinf;
robs        = ((1 + constepinf/100)/((1/(1 + constebeta/100))*(1 + ctrend/100)^(-csigma)) - 1)*100;
labobs      = constelab;
pinfexp4obs = constepinf;
pinflrobs   = constepinf;
end;

steady(solve_algo=1, maxit=1000);

// =========================================================================
// PERFECT FORESIGHT (dynare-rust reference, not in sw07.mod): lines 1-223 of
// reference/sw07/sw07.mod, then temporary productivity and policy-rate
// shocks under perfect foresight.
// =========================================================================
shocks;
var ea; periods 1:2; values 1;
var em; periods 3;   values (-0.5);
end;
perfect_foresight_setup(periods=100);
perfect_foresight_solver;
