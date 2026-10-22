// Dynare's examples/occbin/rbc_occbin.mod (two constraints: INEG adjustment costs below the
// steady-state investment, IRR irreversible investment), with the steady state given by
// initval and STEADY_STATE(Invest) in the model replaced by the parameter Invest_ss
// (dynare-rust does not support STEADY_STATE() inside model equations yet; it does in
// occbin_constraints).

var A C Invest K Lambda log_K log_Invest log_C;
varexo epsilon;
parameters alpha delta beta sigma rho phi psi K_ss Invest_ss;

beta  = 0.96;
alpha = 0.33;
delta = 0.10;
sigma = 2;
rho   = 0.9;
phi   = 0.975;
psi   = 5;
K_ss  = ((1/beta-1+delta)/alpha)^(1/(alpha-1));
Invest_ss = delta*K_ss;

model;
[name='Euler', bind = 'INEG']
-C^(-sigma)*(1+2*psi*(K/K(-1)-1)/K(-1))+ beta*C(+1)^(-sigma)*((1-delta)-2*psi*(K(+1)/K-1)*
  (-K(+1)/K^2)+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

[name='Euler', relax = 'INEG']
-C^(-sigma) + beta*C(+1)^(-sigma)*(1-delta+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

[name='Budget constraint',bind = 'INEG']
C+K-(1-delta)*K(-1)+psi*(K/K(-1)-1)^2=exp(A)*K(-1)^(alpha);

[name='Budget constraint',relax = 'INEG']
C+K-(1-delta)*K(-1)=exp(A)*K(-1)^(alpha);

[name='LOM capital']
Invest = K-(1-delta)*K(-1);

[name='investment',bind='IRR,INEG']
(log_Invest - log(phi*Invest_ss)) = 0;
[name='investment',relax='IRR']
Lambda=0;
[name='investment',bind='IRR',relax='INEG']
(log_Invest - log(phi*Invest_ss)) = 0;

[name='LOM TFP']
A = rho*A(-1)+epsilon;

[name='Definition log capital']
log_K=log(K);
[name='Definition log consumption']
log_C=log(C);
[name='Definition log investment']
log_Invest=log(Invest);
end;

occbin_constraints;
name 'IRR'; bind log_Invest-log(steady_state(Invest))<log(phi); relax Lambda<0;
name 'INEG'; bind log_Invest-log(steady_state(Invest))<-0.000001; %not exactly 0 for numerical reasons
end;

initval;
K = K_ss;
C = -delta*K_ss + K_ss^alpha;
Invest = Invest_ss;
log_K = log(K_ss);
log_C = log(-delta*K_ss + K_ss^alpha);
log_Invest = log(Invest_ss);
Lambda = 0;
A = 0;
end;
steady;

shocks;
  var epsilon; stderr 0.015;
end;

shocks(surprise);
var epsilon;
periods 1:9, 10, 50, 90, 130, 131:169;
values -0.0001, -0.01,-0.02, 0.01, 0.02, 0;
end;

occbin_setup;
@#ifndef VARIANT
occbin_solver(simul_periods=200,simul_check_ahead_periods=200);
@#else
occbin_solver(simul_periods=180, simul_check_ahead_periods=20, simul_curb_retrench, simul_maxit=100);
@#endif
occbin_write_regimes;
occbin_graph log_C epsilon Lambda log_K log_Invest A;
