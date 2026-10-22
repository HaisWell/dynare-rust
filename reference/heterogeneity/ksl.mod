// Krusell-Smith (1998), after examples/heterogeneity/krusell_smith_1998.mod: loads the
// pre-computed steady state krusell_smith_1998.mat (the Va variable is a heterogeneous
// variable with a marginal-value policy; the Euler equation has a borrowing constraint).
heterogeneity_dimension households;

var(heterogeneity=households)
    c  (long_name = 'Consumption')
    a  (long_name = 'Assets')
    Va (long_name = 'Derivative of the value function w.r.t assets')
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
    rho_Z (long_name = 'Aggregate TFP shock persistence')
    sig_Z (long_name = 'Aggregate TFP shock innovation std err')
    Z_ss  (long_name = 'Aggregate TFP shock average value')
;

L = 1;
Z_ss = 0.8816460975214567;
alpha = 0.11;
beta = 0.9819527880123727;
delta = 0.025;
eis = 1;

model(heterogeneity=households);
    [name='Euler equation with borrowing constraint']
    c^(-1/eis) = beta*Va(+1) ⟂ a>=0;

    [name='Budget constraint']
    c + a = (1+r)*a(-1) + w*e;

    [name='Envelope condition']
    Va = (1+r)*c^(-1/eis);
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

heterogeneity_load_steady_state(filename = krusell_smith_1998);
