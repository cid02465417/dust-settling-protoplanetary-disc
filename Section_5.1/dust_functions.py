"""
Disc model and advection-only dust settling solver.

Physical disc / grain properties as functions of cylindrical radius R,
dimensionless vertical profiles (z_hat = z / H, t_hat = Omega t), and a
first-order, mass-conserving upwind (finite-volume) step for the dust density.

Grid convention
---------------
The domain [0, z_max] is split into N cells of width dz. Densities live at the
cell centres z_j = (j + 1/2) dz; fluxes live on the cell faces z = j dz. The
midplane z = 0 is therefore a face, where the flux is exactly zero by symmetry.
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

YEAR   = 365.25 * 24 * 3600.0 # seconds in a year

EPS_0  = 0.01                 # initial dust-to-gas ratio


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

        t_s(z) = sqrt(pi / 8) * (rho_in * a) / (rho_g(z) * c_s)

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
    return np.sqrt(np.pi / 8.0) * (RHO_IN * a) / (rho_g0 * cs)


def St0_from_R_a(R_AU, a):
    """
    Dimensionless stopping time (Stokes number) at the midplane:

        St0 = Omega * t_s(z=0)

    The terminal-velocity approximation used for the settling speed is only
    valid for St << 1.
    """
    return Omega_K(R_AU) * stopping_time_midplane(R_AU, a)


# ------------------------------------------------------------
# Dimensionless vertical grid and profiles
# ------------------------------------------------------------

def make_z_grid(z_max=3.0, N=400):
    """
    Create a uniform cell-centred grid in dimensionless height z_hat.

    The N cells cover [0, z_max]; cell j spans [j dz, (j+1) dz].

    Returns
    -------
    z  : (N,) array of cell-centre heights, z_j = (j + 1/2) dz
    dz : cell width in z_hat
    """
    dz = z_max / N
    z = (np.arange(N) + 0.5) * dz
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


def face_velocity(z_hat, dz_hat, St0):
    """
    Settling velocity on the lower face of every cell.

    z_hat are the cell centres from make_z_grid, so the lower faces sit at
    z_hat - dz_hat / 2 = j dz. The first face is the midplane, where the
    velocity (and so the flux) is exactly zero.
    """
    z_face = z_hat - 0.5 * dz_hat
    return velocity_dimless(z_face, St0)


# ------------------------------------------------------------
# 1D dust advection step (upwind, explicit Euler, finite volume)
# ------------------------------------------------------------

def dust_step(rho, dz, dt, vel_face):
    """
    One explicit, conservative upwind step for the dust density.

    rho      : (N,) array   cell-averaged dust density (dimensionless)
    dz       : float        cell width in z_hat
    dt       : float        time step in t_hat
    vel_face : (N,) array   velocity on the lower face of each cell
                            (from face_velocity), assumed <= 0

    The flux through the lower face of cell j is

        F_j = u(z = j dz) * rho_j

    The dust moves downwards, so the upwind (donor) cell is the one above
    the face, which is cell j itself. The update is

        rho_j^{n+1} = rho_j^n - dt/dz * (F_{j+1} - F_j)

    Boundary conditions:
        z = 0     : reflective. The face velocity is zero there, so F_0 = 0
                    and nothing crosses the midplane.
        z = z_max : no inflow. The flux through the top face is zero because
                    there is no dust above the domain to settle in.

    Every flux leaves one cell and enters its neighbour, so the total dust
    mass sum(rho) * dz is conserved to rounding error.
    """
    F_lower = vel_face * rho                 # F_0 = 0 because vel_face[0] = 0
    F_upper = np.append(F_lower[1:], 0.0)    # top face: no inflow
    return rho - dt / dz * (F_upper - F_lower)


def advection_timestep(dz, vel_face, cfl=0.4):
    """
    Largest stable advection time step, dt = cfl * dz / max|u|  (cfl <= 1).
    """
    return cfl * dz / np.max(np.abs(vel_face))


def dust_to_gas_ratio_dimless(rho_d_hat, z_hat):
    """
    Dust-to-gas ratio epsilon(z_hat) = rho_d / rho_g, given

        rho_d_hat = rho_d / rho_g(R, 0)
        rho_g_hat = rho_g / rho_g(R, 0)

    so epsilon = rho_d_hat / rho_g_hat.
    """
    rho_g_hat = gas_density_dimless(z_hat)
    return rho_d_hat / rho_g_hat


def dust_mass(rho_d_hat, dz_hat):
    """
    Dust column (per unit area, one side of the disc) in dimensionless units.
    Constant in time for a conservative scheme.
    """
    return np.sum(rho_d_hat) * dz_hat


# ------------------------------------------------------------
# Analytic results for the advection-only problem
# ------------------------------------------------------------

def midplane_epsilon_advection(t_hat, St0, eps0=EPS_0):
    """
    Exact midplane dust-to-gas ratio without turbulence.

    At z = 0 the advection equation reduces to
    d(rho_d)/dt = -(du/dz) rho_d = St0 * rho_d, so

        epsilon(z = 0, t_hat) = eps0 * exp(St0 * t_hat)
    """
    return eps0 * np.exp(St0 * t_hat)


def time_to_epsilon_one_advection(St0, eps0=EPS_0):
    """
    Dimensionless time for the midplane to reach epsilon = 1 without
    turbulence:  t_hat = ln(1 / eps0) / St0.
    """
    return np.log(1.0 / eps0) / St0
