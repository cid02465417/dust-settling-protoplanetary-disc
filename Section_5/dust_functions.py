"""
Disc model and advection-only dust settling solver.

Physical disc / grain properties as functions of cylindrical radius R,
dimensionless vertical profiles (z_hat = z / H, t_hat = Omega t), and a
first-order upwind step for the dust density.
"""

import numpy as np


# ------------------------------------------------------------
# Physical constants & disc / grain parameters (SI units)
# ------------------------------------------------------------

G      = 6.67430e-11          # gravitational constant
M_STAR = 2.0e30               # stellar mass (~1 Msun) in kg
AU     = 1.5e11               # 1 AU in m

k_B    = 1.380649e-23         # Boltzmann constant
m_H    = 1.6735575e-27        # hydrogen mass
RHO_IN = 1000.0               # internal grain density rho_in


# ------------------------------------------------------------
# Functions of cylindrical radius R (in AU)
# ------------------------------------------------------------

def temperature_R(R_AU):
    """
    Disc midplane temperature T(R) in Kelvin.

        T(R) = 264 K * (R / 1 AU)^(-1/2)
    """
    return 264.0 * R_AU**(-0.5)


def rho_g0_R(R_AU):
    """
    Midplane gas density rho_g(R, z=0) in kg/m^3.

        rho_g(R, 0) = 5e-7 kg/m^3 * (R / 1 AU)^(-3)
    """
    return 5.0e-7 * R_AU**(-3.0)


def sound_speed(R_AU):
    """
    Gas sound speed c_s(R) in m/s.

        c_s = sqrt( k_B T(R) / (2 m_H) )

    (Project definition; the factor 2 in the denominator is given.)
    """
    T = temperature_R(R_AU)
    return np.sqrt(k_B * T / (2.0 * m_H))


def Omega_K(R_AU):
    """
    Keplerian angular frequency Omega(R) in s^-1.

        Omega = sqrt( G M / R^3 ),  with R in metres.
    """
    R_m = R_AU * AU
    return np.sqrt(G * M_STAR / R_m**3)


def scale_height(R_AU):
    """
    Disc scale height H(R) in metres.

        H = c_s / Omega
    """
    return sound_speed(R_AU) / Omega_K(R_AU)


# ------------------------------------------------------------
# Stopping time & Stokes number at the midplane
# ------------------------------------------------------------

def stopping_time_midplane(R_AU, a):
    """
    Stopping time t_s at the midplane z = 0, in seconds.

    Project formula (Epstein drag):

        t_s(z) = (sqrt(pi) / 8) * (rho_in * a) / (rho_g(z) * c_s)

    evaluated at z = 0, so rho_g = rho_g(R, 0).

    Parameters
    ----------
    R_AU : float
        Cylindrical radius in AU.
    a : float
        Grain size in metres.
    """
    rho_g0 = rho_g0_R(R_AU)
    cs     = sound_speed(R_AU)
    return np.sqrt(np.pi) / 8.0 * (RHO_IN * a) / (rho_g0 * cs)


def St0_from_R_a(R_AU, a):
    """
    Dimensionless stopping time (Stokes number) at the midplane:

        St0 = Omega * t_s(z=0)
    """
    return Omega_K(R_AU) * stopping_time_midplane(R_AU, a)


# ------------------------------------------------------------
# Dimensionless vertical grid and profiles
# ------------------------------------------------------------

def make_z_grid(z_max=5.0, N=400):
    """
    Create a uniform grid in dimensionless height z_hat on [0, z_max).

    Returns
    -------
    z  : (N,) array of z_hat values
    dz : grid spacing in z_hat
    """
    dz = z_max / N
    z = np.linspace(0.0, z_max - dz, N)   # half-open interval
    return z, dz


def gas_density_dimless(z_hat):
    """
    Dimensionless gas density profile:

        rho_g_hat(z_hat) = rho_g(z) / rho_g(R, 0) = exp(-z_hat^2 / 2)
    """
    return np.exp(-0.5 * z_hat**2)


def ts_dimless(z_hat, St0):
    """
    Dimensionless stopping time:

        t_s_hat(z_hat) = Omega t_s(z) = St0 / rho_g_hat(z_hat)

    since t_s is proportional to 1 / rho_g.
    """
    return St0 / gas_density_dimless(z_hat)


def velocity_dimless(z_hat, St0):
    """
    Dimensionless settling velocity.

    From u_z = -Omega^2 z t_s(z), with z_hat = z / H, t_hat = Omega t
    and u_hat = u_z / (H Omega):

        u_hat(z_hat) = -z_hat * t_s_hat(z_hat)

    Negative: dust settles towards z = 0.
    """
    return -z_hat * ts_dimless(z_hat, St0)


# ------------------------------------------------------------
# 1D dust advection step (upwind, explicit Euler)
# ------------------------------------------------------------

def dust_step(rho, dz, dt, vel):
    """
    One explicit upwind step for the dust density with variable velocity.

    rho : (N,) array   dust density (dimensionless) at the current time
    dz  : float        grid spacing in z_hat
    dt  : float        time step in t_hat
    vel : (N,) array   dimensionless velocity u_hat(z_hat), assumed <= 0

    Boundary conditions:
        z = 0     : reflective (no-flux) via a mirrored ghost cell
        z = z_max : outflow, rho_ghost = 0
    """
    # Ghost cell at the midplane (reflective BC)
    rho = np.insert(rho, 0, rho[1])
    vel = np.insert(vel, 0, -vel[1])

    # Ghost cell at the top (outflow BC)
    rho = np.append(rho, 0.0)
    vel = np.append(vel, 0.0)

    rho_new = rho.copy()

    # Flux
    F = vel * rho

    # Interior update
    rho_new[1:-1] -= dt / dz * (F[2:] - F[1:-1])

    # Extra correction at the midplane using ghost + first cell
    rho_new[1] -= dt / dz * (F[1] - F[0])

    # Strip ghost cells before returning
    return rho_new[1:-1]


def dust_to_gas_ratio_dimless(rho_d_hat, z_hat):
    """
    Dust-to-gas ratio epsilon(z_hat) = rho_d / rho_g, given

        rho_d_hat = rho_d / rho_g(R, 0)
        rho_g_hat = rho_g / rho_g(R, 0)

    so epsilon = rho_d_hat / rho_g_hat.
    """
    rho_g_hat = gas_density_dimless(z_hat)
    return rho_d_hat / (rho_g_hat + 1e-300)
