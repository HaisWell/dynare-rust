/*
 * Test functionality of model_info on a more complex model with many leads and lags/states.
 * Prevents regression where loop over states did lead to index overflow of block indices.
 */

/*
 * Copyright © 2010-2026 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
 */

@#define countries = [ "H", "F" ]
@#define J = 4

@#for co in countries
var C_@{co} L_@{co} N_@{co} A_@{co} K_@{co} Z_@{co} X_@{co} LAMBDA_@{co} S_@{co} NX_@{co} Y_@{co};

varexo E_@{co};

parameters beta_@{co} alpha_@{co} eta_@{co} mu_@{co} gamma_@{co} theta_@{co} nu_@{co} sigma_@{co} delta_@{co} phi_@{co} psi_@{co} rho_@{co}_@{co};
@#endfor

// Lagrange multiplier of aggregate constraint
var LGM;

parameters rho_@{countries[1]}_@{countries[2]} rho_@{countries[2]}_@{countries[1]};

model(block);
@#for co in countries

Y_@{co} = ((LAMBDA_@{co}*K_@{co}(-@{J})^theta_@{co}*N_@{co}^(1-theta_@{co}))^(-nu_@{co}) + sigma_@{co}*Z_@{co}(-1)^(-nu_@{co}))^(-1/nu_@{co});
K_@{co} = (1-delta_@{co})*K_@{co}(-1) + S_@{co};
X_@{co} =
@# for lag in (-J+1):0
          + phi_@{co}*S_@{co}(@{lag})
@# endfor
;

A_@{co} = (1-eta_@{co})*A_@{co}(-1) + N_@{co};
L_@{co} = 1 - alpha_@{co}*N_@{co} - (1-alpha_@{co})*eta_@{co}*A_@{co}(-1);

// Utility multiplied by gamma
# U_@{co} = (C_@{co}^mu_@{co}*L_@{co}^(1-mu_@{co}))^gamma_@{co};

// FOC with respect to consumption
psi_@{co}*mu_@{co}/C_@{co}*U_@{co} = LGM;

// FOC with respect to labor
// NOTE: this condition is only valid for alpha = 1
psi_@{co}*(1-mu_@{co})/L_@{co}*U_@{co}*(-alpha_@{co}) = - LGM * (1-theta_@{co})/N_@{co}*(LAMBDA_@{co}*K_@{co}(-@{J})^theta_@{co}*N_@{co}^(1-theta_@{co}))^(-nu_@{co})*Y_@{co}^(1+nu_@{co});

// FOC with respect to capital
@# for lag in 0:(J-1)
 +beta_@{co}^@{lag}*LGM(+@{lag})*phi_@{co}
@# endfor
@# for lag in 1:J
 -beta_@{co}^@{lag}*LGM(+@{lag})*phi_@{co}*(1-delta_@{co})
@# endfor
 = beta_@{co}^@{J}*LGM(+@{J})*theta_@{co}/K_@{co}*(LAMBDA_@{co}(+@{J})*K_@{co}^theta_@{co}*N_@{co}(+@{J})^(1-theta_@{co}))^(-nu_@{co})*Y_@{co}(+@{J})^(1+nu_@{co});

// FOC with respect to stock of inventories
 LGM=beta_@{co}*LGM(+1)*(1+sigma_@{co}*Z_@{co}^(-nu_@{co}-1)*Y_@{co}(+1)^(1+nu_@{co}));

// Shock process
@# if co == countries[1]
@#  define alt_co = countries[2]
@# else
@#  define alt_co = countries[1]
@# endif
 (LAMBDA_@{co}-1) = rho_@{co}_@{co}*(LAMBDA_@{co}(-1)-1) + rho_@{co}_@{alt_co}*(LAMBDA_@{alt_co}(-1)-1) + E_@{co};


NX_@{co} = (Y_@{co} - (C_@{co} + X_@{co} + Z_@{co} - Z_@{co}(-1)))/Y_@{co};

@#endfor

// World ressource constraint
@#for co in countries
  +C_@{co} + X_@{co} + Z_@{co} - Z_@{co}(-1)
@#endfor
    =
@#for co in countries
  +Y_@{co}
@#endfor
    ;

end;

@#for co in countries
beta_@{co} = 0.99;
mu_@{co} = 0.34;
gamma_@{co} = -1.0;
alpha_@{co} = 1;
eta_@{co} = 0.5; // Irrelevant when alpha=1
theta_@{co} = 0.36;
nu_@{co} = 3;
sigma_@{co} = 0.01;
delta_@{co} = 0.025;
phi_@{co} = 1/@{J};
psi_@{co} = 0.5;
@#endfor

rho_H_H = 0.906;
rho_F_F = 0.906;
rho_H_F = 0.088;
rho_F_H = 0.088;

initval;
@#for co in countries
LAMBDA_@{co} = 1;
NX_@{co} = 0;
Z_@{co} = 1;
A_@{co} = 1;
L_@{co} = 0.5;
N_@{co} = 0.5;
Y_@{co} = 1;
K_@{co} = 1;
C_@{co} = 1;
S_@{co} = 1;
X_@{co} = 1;

E_@{co} = 0;
@#endfor

LGM = 1;
end;

shocks;
var E_H; stderr 0.00852;
var E_F; stderr 0.00852;
corr E_H, E_F = 0.258;
end;



model_info(block_static);
model_info(block_static, incidence);
model_info(block_dynamic);
model_info(block_dynamic, incidence);
