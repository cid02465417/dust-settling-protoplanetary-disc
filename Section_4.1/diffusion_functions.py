"""
Turbulent diffusion term for dust settling, combined with the upwind
advection step from dust_functions, plus the analytic steady state.

Uses the same cell-centred grid as dust_functions: densities at cell centres,
fluxes on cell faces, zero flux through the midplane and the top of the domain.
"""

import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2
_trapz = getattr(np, "trapezoid", None) or np.trapz

from dust_functions import (
    gas_density_dimless, dust_step, St0_from_R_a, EPS_0,
)


# ------------------------------------------------------------
# Diffusion RHS
# ------------------------------------------------------------

def diffusion_rhs_dimless(rho_d_hat, z_hat, alpha):
    """
    RHS of the turbulent diffusion term in dimensionless form:

        d(rho_d_hat)/d(t_hat) |_diff
            = alpha * d/dz_hat [ rho_g_hat * d/dz_hat ( rho_d_hat / rho_g_hat ) ]

    The diffusive flux on the face between cells j-1 and j is

        F_j = -alpha * rho_g(z_face) * (f_j - f_{j-1}) / dz,   f = rho_d / rho_g

    with zero flux on the midplane face and on the top face, so the total
    dust mass is conserved.
    """
    if alpha <= 0.0:
        return np.zeros_like(rho_d_hat)

    dz_hat = z_hat[1] - z_hat[0]

    # f = rho_d / rho_g at cell centres
    f = rho_d_hat / gas_density_dimless(z_hat)

    # Interior faces sit half a cell below centres 1..N-1
    z_face = z_hat[1:] - 0.5 * dz_hat
    rho_g_face = gas_density_dimless(z_face)

    # Fluxes on all N+1 faces; the two boundary faces stay at zero
    F = np.zeros(len(rho_d_hat) + 1)
    F[1:-1] = -alpha * rho_g_face * (f[1:] - f[:-1]) / dz_hat

    # Divergence of flux
    return -(F[1:] - F[:-1]) / dz_hat


# ------------------------------------------------------------
# Advection + diffusion combined step
# ------------------------------------------------------------

def dust_step_with_diffusion(rho_d_hat, z_hat, dt_hat, vel_face, alpha):
    """
    One explicit step: upwind advection followed by turbulent diffusion.

    vel_face is the settling velocity on the lower cell faces
    (dust_functions.face_velocity).
    """
    dz_hat = z_hat[1] - z_hat[0]

    # 1) Advection-only step
    rho_adv = dust_step(rho_d_hat, dz_hat, dt_hat, vel_face)

    # 2) Diffusion RHS
    rhs_diff = diffusion_rhs_dimless(rho_adv, z_hat, alpha)

    # 3) Update
    return rho_adv + dt_hat * rhs_diff


# ------------------------------------------------------------
# Stable timestep
# ------------------------------------------------------------

def max_dt_hat_for_diffusion(dz_hat, alpha, cfl=0.4):
    """
    Explicit diffusion stability condition:

        dt_hat <= cfl * dz_hat^2 / (2 * alpha)
    """
    if alpha <= 0.0:
        return np.inf

    return cfl * dz_hat**2 / (2.0 * alpha)


def stable_timestep(dz_hat, vel_face, alpha, cfl=0.4):
    """
    Time step satisfying both the advection (CFL) and diffusion limits.
    """
    dt_adv = cfl * dz_hat / np.max(np.abs(vel_face))
    dt_diff = max_dt_hat_for_diffusion(dz_hat, alpha, cfl=cfl)
    return min(dt_adv, dt_diff)


def response_time(St0, alpha):
    """
    Dimensionless time on which the dust layer relaxes towards its steady
    state, 1 / (St0 + alpha). A run needs ~10 of these to be converged.
    """
    return 1.0 / (St0 + alpha)


# ------------------------------------------------------------
# Analytic steady state (settling balanced by diffusion)
# ------------------------------------------------------------

def steady_state_profile(z_hat, St0, alpha, eps0=EPS_0, z_max=3.0, n_quad=200001):
    """
    Analytic steady-state dust density rho_d_hat(z_hat) for alpha > 0.

    Setting the total flux to zero,
        u rho_d = alpha rho_g d(rho_d / rho_g)/dz,
    with u = -z St0 exp(z^2/2) gives

        rho_d / rho_g = eps_mid * exp( -(St0/alpha) * (exp(z^2/2) - 1) )

    (the standard constant-alpha result; see e.g. Dubrulle et al. 1995,
    Fromang & Nelson 2009). eps_mid is fixed by conserving
    the initial dust mass, eps0 * integral of rho_g, inside [0, z_max].
    """
    S = St0 / alpha

    def shape(z):
        return np.exp(-S * np.expm1(0.5 * z**2) - 0.5 * z**2)

    zq = np.linspace(0.0, z_max, n_quad)
    dust_integral = _trapz(shape(zq), zq)
    gas_integral = _trapz(gas_density_dimless(zq), zq)
    eps_mid = eps0 * gas_integral / dust_integral

    return eps_mid * shape(np.asarray(z_hat, dtype=float))


def steady_state_midplane_epsilon(St0, alpha, eps0=EPS_0, z_max=3.0):
    """
    Analytic steady-state midplane dust-to-gas ratio (z = 0).

    For St0 >> alpha this tends to eps0 * sqrt(St0 / alpha) times a factor
    close to one, so epsilon > 1 needs St0 / alpha of order 1 / eps0^2 = 10^4.
    """
    return float(steady_state_profile(0.0, St0, alpha, eps0=eps0, z_max=z_max))


def threshold_grain_size(R_AU, alpha, eps_target=1.0, eps0=EPS_0, z_max=3.0):
    """
    Grain size (metres) at which the analytic steady-state midplane
    dust-to-gas ratio equals eps_target, found by bisection in log(a).

    St0 is proportional to a, so the answer is a = St0_crit / St0(a = 1 m).
    """
    St0_per_metre = St0_from_R_a(R_AU, 1.0)

    lo, hi = -9.0, 3.0   # log10(a / m)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eps = steady_state_midplane_epsilon(St0_per_metre * 10.0**mid, alpha,
                                            eps0=eps0, z_max=z_max)
        if eps < eps_target:
            lo = mid
        else:
            hi = mid
    return 10.0**(0.5 * (lo + hi))
