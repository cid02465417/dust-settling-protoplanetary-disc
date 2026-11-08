"""
Resolution study: how the number of grid points N affects the
advection + diffusion solution at fixed radius, grain size and alpha.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from dust_functions import (
    make_z_grid,
    St0_from_R_a,
    velocity_dimless,
    gas_density_dimless,
    Omega_K
)

from diffusion_functions import (
    dust_step_with_diffusion,
    max_dt_hat_for_diffusion,
)


# Parameters

R_AU  = 30.0                 # radius (AU)
alpha = 4e-4                 # turbulence strength
a_m   = 1e-4                 # grain size (m)
z_max = 3.0
N_values = [200, 400, 800]   # grid resolutions to compare

year = 365.25 * 24 * 3600.0  # seconds in a year
Omega_R = Omega_K(R_AU)
t_hat_final = 1e4 * year * Omega_R   # final dimensionless time (10^4 years)


plt.figure(figsize=(6, 5))

midplane_eps = []

for N_z in N_values:

    # Grid and gas profile
    z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)
    rho_g_hat = gas_density_dimless(z_hat)

    start = time.time()
    print("\n" + "-" * 60)
    print(f"N = {N_z}, grain size a = {a_m:.2e} m ({a_m*1e6:.1f} µm)")

    # Initial dust (well mixed), stopping time, velocity
    rho_d_hat = 0.01 * rho_g_hat.copy()
    St0 = St0_from_R_a(R_AU, a_m)
    u_hat = velocity_dimless(z_hat, St0)

    # dt_hat from advection + diffusion CFL
    umax = np.max(np.abs(u_hat))
    dt_adv = 0.4 * dz_hat / umax
    dt_diff = max_dt_hat_for_diffusion(dz_hat, alpha, cfl=0.4)
    dt_hat = min(dt_adv, dt_diff)       # choose the smaller one for a safe step

    print(f"Estimated steps ≈ {t_hat_final / dt_hat:.0f}")
    print(f"t_final ≈ {t_hat_final / Omega_R / year:.3e} years")

    # Time evolution
    t_hat = 0.0
    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(
            rho_d_hat, z_hat, dt_hat, u_hat, alpha
        )
        t_hat += dt_hat

    print(f"Wall-clock time: {time.time() - start:.2f} seconds")

    epsilon_final = rho_d_hat / rho_g_hat
    midplane_eps.append(epsilon_final[0])
    print(f"Midplane epsilon (R = {R_AU:.0f} AU, alpha = {alpha:.2e}) = {epsilon_final[0]:.3e}")

    # Plot final epsilon(z) profile for this resolution
    plt.plot(z_hat, epsilon_final, label=f"N = {N_z}")


plt.xlabel("z / H")
plt.ylabel("epsilon = rho_d / rho_g")
plt.title(f"Dust-to-gas ratio at R = {R_AU:.0f} AU, alpha = {alpha}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Midplane epsilon against resolution
plt.figure()
plt.plot(N_values, midplane_eps, marker="o")
plt.xlabel("Grid points, N")
plt.ylabel("epsilon = rho_d / rho_g")
plt.title("Midplane epsilon against N")
plt.grid(True)
plt.tight_layout()
plt.show()
