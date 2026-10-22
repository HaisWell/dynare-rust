// Krusell-Smith (1998), after examples/heterogeneity/krusell_smith_1998_steady_state.mod:
// steady state computed from an initial guess stored in ks_guess.mat (written by dump.m),
// with the discount factor beta calibrated so that capital market clearing holds.
heterogeneity_dimension households;

var(heterogeneity=households)
    c  (long_name = 'Consumption')
    a  (long_name = 'Assets')
;

varexo(heterogeneity=households)
    e  (long_name = 'Idiosyncratic productivity shock')
;

var
    Y (long_name = 'Aggregate output')
    r (long_name = 'Rate of return on capital net of depreciation')
    w (long_name = 'Wage rate')
    K (long_name = 'Aggregate capital')
;

varexo Z (long_name = 'Aggregate productivity shock');

parameters
    L     (long_name = 'Labor')
    alpha (long_name = 'Share of capital in production function')
    beta  (long_name = 'Subjective discount rate of households')
    delta (long_name = 'Capital depreciation rate')
    eis   (long_name = 'Elasticity of intertemporal substitution')
    Z_ss  (long_name = 'Aggregate TFP shock average value')
;

delta = 0.025;
eis = 1;
L = 1;
alpha = 0.11;
beta = 0.98;
Z_ss = 0.8816460975214567;

model(heterogeneity=households);
    [name='Euler equation with borrowing constraint']
    c^(-1/eis) = beta*(1+r(+1))*c(+1)^(-1/eis) ⟂ a>=0;

    [name='Budget constraint']
    c + a = (1+r)*a(-1) + w*e;
end;

model;
    [name='Production function']
    Y = (Z_ss+Z) * K(-1)^alpha * L^(1 - alpha);

    [name='Capital rental rate']
    r = alpha * (Z_ss+Z) * (K(-1) / L)^(alpha - 1) - delta;

    [name='Wage rate']
    w = (1 - alpha) * (Z_ss+Z) * (K(-1) / L)^alpha;

    [name='Capital market clearing']
    K = SUM(a);
end;

shocks;
    var Z; stderr 0.01;
end;

shocks(heterogeneity=households);
    var e; stderr 0.2;
end;

heterogeneity_compute_steady_state(filename = ks_guess, variable = initial_guess,
    time_iteration_verbosity = 0, forward_verbosity = 0, calibration_verbosity = 1);
