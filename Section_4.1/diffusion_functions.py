"""
Turbulent diffusion term for dust settling, combined with the upwind
advection step from dust_functions.
"""

import numpy as np

from dust_functions import gas_density_dimless, dust_step


# ------------------------------------------------------------
# Diffusion RHS
# ------------------------------------------------------------

def diffusion_rhs_dimless(rho_d_hat, z_hat, alpha):
    """
    RHS of the turbulent diffusion term in dimensionless form:

        d(rho_d_hat)/d(t_hat) |_diff
            = alpha * d/dz_hat [ rho_g_hat * d/dz_hat ( rho_d_hat / rho_g_hat ) ]

    Zero-flux boundaries at the midplane and at the top of the domain.
    """
    if alpha <= 0.0:
        return np.zeros_like(rho_d_hat)

    rhs_diff = np.zeros_like(rho_d_hat)

    dz_hat = z_hat[1] - z_hat[0]

    # Gas density profile
    rho_g_hat = gas_density_dimless(z_hat)

    # f = rho_d / rho_g
    f = rho_d_hat / (rho_g_hat + 1e-300)

    # Gas density at cell faces
    rho_g_face = 0.5 * (rho_g_hat[1:] + rho_g_hat[:-1])

    # Gradient of f at faces
    df_dz_face = (f[1:] - f[:-1]) / dz_hat

    # Diffusive flux at faces
    F_face = -alpha * rho_g_face * df_dz_face

    # Divergence of flux
    rhs_diff[1:-1] = -(F_face[1:] - F_face[:-1]) / dz_hat

    rhs_diff[-1] = -(0.0 - F_face[-1]) / dz_hat    # BC at top
    rhs_diff[0] = -(F_face[0] - 0.0) / dz_hat      # BC at midplane

    return rhs_diff


# ------------------------------------------------------------
# Advection + diffusion combined step
# ------------------------------------------------------------

def dust_step_with_diffusion(rho_d_hat, z_hat, dt_hat, vel_hat, alpha):
    """
    One explicit step: upwind advection followed by turbulent diffusion.
    """
    dz_hat = z_hat[1] - z_hat[0]

    # 1) Advection-only step
    rho_adv = dust_step(rho_d_hat, dz_hat, dt_hat, vel_hat)

    # 2) Diffusion RHS
    rhs_diff = diffusion_rhs_dimless(rho_adv, z_hat, alpha)

    # 3) Update
    return rho_adv + dt_hat * rhs_diff


# ------------------------------------------------------------
# Stable timestep for diffusion
# ------------------------------------------------------------

def max_dt_hat_for_diffusion(dz_hat, alpha, cfl=0.4):
    """
    Explicit diffusion stability condition:

        dt_hat <= cfl * dz_hat^2 / (2 * alpha)
    """
    if alpha <= 0.0:
        return np.inf

    return cfl * dz_hat**2 / (2.0 * alpha)
