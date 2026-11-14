"""
Checks of the dust solver against exact results.

1. Mass conservation of the advection + diffusion step.
2. Advection only: the midplane dust-to-gas ratio must follow
   epsilon(0, t) = 0.01 * exp(St0 * t), with first-order convergence in dz.
3. Advection + diffusion: the steady state must match the analytic profile
   rho_d / rho_g = eps_mid * exp(-(St0/alpha) * (exp(z^2/2) - 1)).

Runs in about a minute and prints PASS / FAIL for each check.
"""

import numpy as np

from dust_functions import (
    make_z_grid, gas_density_dimless, face_velocity, dust_step,
    advection_timestep, dust_mass, midplane_epsilon_advection, EPS_0,
)
from diffusion_functions import (
    dust_step_with_diffusion, stable_timestep, response_time,
    steady_state_profile,
)

Z_MAX = 3.0
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ------------------------------------------------------------
# 1. Mass conservation
# ------------------------------------------------------------

St0, alpha, N = 1.0e-2, 1.0e-3, 200
z, dz = make_z_grid(Z_MAX, N)
rho_g = gas_density_dimless(z)
u = face_velocity(z, dz, St0)
dt = stable_timestep(dz, u, alpha)

rho = EPS_0 * rho_g
m0 = dust_mass(rho, dz)
for _ in range(50000):
    rho = dust_step_with_diffusion(rho, z, dt, u, alpha)
drift = abs(dust_mass(rho, dz) / m0 - 1.0)
report("mass conservation", drift < 1e-10,
       f"relative change after 50000 steps = {drift:.1e}")
report("positivity", rho.min() >= 0.0, f"min rho_d = {rho.min():.2e}")


# ------------------------------------------------------------
# 2. Advection only: exponential growth at the midplane
# ------------------------------------------------------------

print("\nAdvection only, epsilon at the lowest cell after t_hat = 2 / St0")
St0 = 1.0e-3
t_end = 2.0 / St0
errors, dzs = [], []
for N in (100, 200, 400, 800):
    z, dz = make_z_grid(Z_MAX, N)
    rho_g = gas_density_dimless(z)
    u = face_velocity(z, dz, St0)
    dt = advection_timestep(dz, u)
    rho = EPS_0 * rho_g
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps                     # land exactly on t_end
    for _ in range(n_steps):
        rho = dust_step(rho, dz, dt, u)
    eps_num = rho[0] / rho_g[0]
    eps_exact = midplane_epsilon_advection(t_end, St0)
    err = abs(eps_num / eps_exact - 1.0)
    errors.append(err)
    dzs.append(dz)
    print(f"  N = {N:4d}: numerical {eps_num:.5f}, exact {eps_exact:.5f}, "
          f"relative error {err:.2e}")
order = np.polyfit(np.log(dzs), np.log(errors), 1)[0]
report("midplane growth", errors[-1] < 0.02,
       f"error at N = 800 is {errors[-1]:.2e}")
report("order of convergence", order > 0.7,
       f"measured order = {order:.2f} (at least first order is required)")


# ------------------------------------------------------------
# 3. Advection + diffusion: analytic steady state
# ------------------------------------------------------------

print("\nSteady state with St0 / alpha = 10")
St0, alpha = 1.0e-2, 1.0e-3
errors, dzs = [], []
for N in (100, 200):
    z, dz = make_z_grid(Z_MAX, N)
    rho_g = gas_density_dimless(z)
    u = face_velocity(z, dz, St0)
    dt = stable_timestep(dz, u, alpha)
    rho = EPS_0 * rho_g
    t, t_end = 0.0, 15.0 * response_time(St0, alpha)
    while t < t_end:
        rho = dust_step_with_diffusion(rho, z, dt, u, alpha)
        t += dt
    exact = steady_state_profile(z, St0, alpha, z_max=Z_MAX)
    err = np.max(np.abs(rho - exact)) / exact.max()
    errors.append(err)
    dzs.append(dz)
    print(f"  N = {N:4d}: midplane eps numerical {rho[0]/rho_g[0]:.5f}, "
          f"analytic {exact[0]/rho_g[0]:.5f}, max profile error {err:.2e}")
report("steady state", errors[-1] < 0.03,
       f"max error at N = 200 is {errors[-1]:.2e} of the peak")
report("steady state converges", errors[-1] < errors[0],
       "error falls when the grid is refined")

print("\nAll checks passed." if all(results) else "\nSome checks FAILED.")
