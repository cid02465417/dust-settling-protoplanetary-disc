"""
Resolution study: how the number of grid points N affects the steady-state
advection + diffusion solution, compared with the analytic steady state.

Any difference from the analytic midplane epsilon is numerical error, which
is dominated by the numerical diffusion of the first-order upwind scheme.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from dust_functions import (
    make_z_grid,
    St0_from_R_a,
    face_velocity,
    gas_density_dimless,
    Omega_K,
    YEAR,
)

from diffusion_functions import (
    dust_step_with_diffusion,
    stable_timestep,
    response_time,
    steady_state_profile,
)


# Parameters

R_AU  = 30.0                 # radius (AU)
alpha = 1e-4                 # turbulence strength (value estimated in Section 4.1)
a_m   = 1e-4                 # grain size (m)
z_max = 3.0
N_values = [50, 100, 200, 400]   # grid resolutions to compare

Omega_R = Omega_K(R_AU)
St0 = St0_from_R_a(R_AU, a_m)
t_hat_final = 15.0 * response_time(St0, alpha)   # long enough for a steady state

print(f"R = {R_AU:.0f} AU, a = {a_m*1e6:.0f} µm, alpha = {alpha:.0e}, St0 = {St0:.3e}")
print(f"t_final = {t_hat_final / Omega_R / YEAR:.3e} years (15 response times)")

plt.figure(figsize=(6, 5))

midplane_eps = []
dz_values = []

for N_z in N_values:

    z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)
    rho_g_hat = gas_density_dimless(z_hat)
    u_hat = face_velocity(z_hat, dz_hat, St0)
    dt_hat = stable_timestep(dz_hat, u_hat, alpha)

    start = time.time()
    print("\n" + "-" * 60)
    print(f"N = {N_z}: {t_hat_final / dt_hat:.0f} steps")

    rho_d_hat = 0.01 * rho_g_hat
    t_hat = 0.0
    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(rho_d_hat, z_hat, dt_hat, u_hat, alpha)
        t_hat += dt_hat

    print(f"Wall-clock time: {time.time() - start:.1f} seconds")

    epsilon_final = rho_d_hat / rho_g_hat
    # extrapolate the two lowest cells to z = 0 for a like-for-like comparison
    eps_mid = 1.5 * epsilon_final[0] - 0.5 * epsilon_final[1]
    midplane_eps.append(eps_mid)
    dz_values.append(dz_hat)
    print(f"Midplane epsilon = {eps_mid:.5f}")

    plt.plot(z_hat, epsilon_final, label=f"N = {N_z}")

z_fine = np.linspace(0.0, z_max, 2000)
eps_exact_profile = (steady_state_profile(z_fine, St0, alpha, z_max=z_max)
                     / gas_density_dimless(z_fine))
eps_exact = eps_exact_profile[0]

plt.plot(z_fine, eps_exact_profile, "k:", label="analytic steady state")
plt.xlim(0, 1.0)
plt.xlabel("z / H")
plt.ylabel("epsilon = rho_d / rho_g")
plt.title(f"Dust-to-gas ratio at R = {R_AU:.0f} AU, alpha = {alpha:.0e}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Error in the midplane epsilon against resolution
errors = np.abs(np.array(midplane_eps) / eps_exact - 1.0)
order = np.polyfit(np.log(dz_values), np.log(errors), 1)[0]

print(f"\nAnalytic midplane epsilon = {eps_exact:.5f}")
for N_z, eps, err in zip(N_values, midplane_eps, errors):
    print(f"N = {N_z:4d}: epsilon = {eps:.5f}, relative error = {err:.2e}")
print(f"Measured order of convergence ≈ {order:.2f}")

plt.figure()
plt.loglog(dz_values, errors, marker="o")
plt.gca().invert_xaxis()
plt.xlabel(r"cell width $\Delta \hat z$")
plt.ylabel("relative error in midplane epsilon")
plt.title(f"Convergence to the analytic steady state (order ≈ {order:.2f})")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
