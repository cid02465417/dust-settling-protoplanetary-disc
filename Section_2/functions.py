"""
Section 2 upwind solver (constant v, periodic BC)


We solve the linear advection equation u_t + v u_x = 0 with known (here constant) v.

We do this using a finite-difference upwind method.

We start with the initial condition u_0(x) = exp(-x^2 / L^2),
a Gaussian profile for which we know the analytic solution.
"""


import numpy as np


# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------


def upwind_step(u, dx, dt, v=1.0):
    """
    Perform one upwind time-step with periodic boundary conditions.
    u : array of shape (N,)
    v : advection velocity
    """
    a = v * dt / dx
    if v > 0:
        # u_j^{n+1} = u_j^n - a (u_j^n - u_{j-1}^n)
        u_left = np.roll(u, 1)     # u_{j-1}
        u_new = u - a * (u - u_left)
    else:
        # u_j^{n+1} = u_j^n - a (u_{j+1}^n - u_j^n)
        u_right = np.roll(u, -1)   # u_{j+1}
        u_new = u - a * (u_right - u)
    return u_new


def run_advection_once(N, T_final, step_func, init_func,
                       v=1.0, x_min=-5.0, x_max=5.0,
                       cfl=0.5, L_gauss=1.0):
    """
    General 1D solver loop for constant-velocity advection.

    N         : number of grid points
    T_final   : final time
    step_func : function(u, dx, dt, **params) -> u_new
    init_func : function(x) -> u_initial
    v         : constant advection speed
    x_min/max : domain
    cfl       : CFL number (<= 1 for upwind)
    """
    # Grid
    Lx = x_max - x_min
    dx = Lx / N
    x = np.linspace(x_min, x_max - dx, N)

    # Initial condition
    u = init_func(x)

    # Time step from CFL
    dt = cfl * dx / abs(v)

    # Time integration
    t = 0.0
    while t < T_final:
        u = step_func(u, dx, dt, v=v)
        t += dt

    return x, u, dx, t


def analytic_solution_gaussian(x, t, v, L_gauss, x_min, x_max):
    """
    Analytic solution u(x, t) = exp(- (x - vt)^2 / L^2),
    with periodic wrapping onto [x_min, x_max).
    """
    Lx = x_max - x_min
    x0 = x - v * t
    x0_wrapped = ((x0 - x_min) % Lx) + x_min

    return np.exp(-(x0_wrapped**2) / L_gauss**2)
