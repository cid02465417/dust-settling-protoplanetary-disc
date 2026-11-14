"""
Section 4.1: estimating the turbulence strength alpha at 100 AU.

Evolves 1 micron dust for 1 Myr for several values of alpha and compares the
final profiles with the advection-only case and with the analytic steady state.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from dust_functions import (
    make_z_grid,
    St0_from_R_a,
    face_velocity,
    gas_density_dimless,
    dust_mass,
    Omega_K,
    YEAR,
)

from diffusion_functions import (
    dust_step_with_diffusion,
    stable_timestep,
    steady_state_profile,
)


# Parameters

R_AU = 100.0
a_m  = 1e-6
alphas = [0, 1e-5, 1e-4, 1e-3]   # turbulence strengths to compare

z_max = 3.0
N_z = 400
t_final_years = 1.0e6            # disc lifetime

Omega_R = Omega_K(R_AU)
t_hat_final = t_final_years * YEAR * Omega_R

z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)
rho_g_hat = gas_density_dimless(z_hat)
St0 = St0_from_R_a(R_AU, a_m)
u_hat = face_velocity(z_hat, dz_hat, St0)

# index of the cell containing z = 1 H (the observed dust height, z ~ 0.1 R)
i_1H = np.searchsorted(z_hat, 1.0)

print(f"R = {R_AU:.0f} AU, a = {a_m*1e6:.1f} µm, St0 = {St0:.3e}")
print(f"t_final = {t_final_years:.1e} years = {t_hat_final:.0f} / Omega")

plt.figure(figsize=(6.5, 5))

for alpha in alphas:
    rho_d_hat = 0.01 * rho_g_hat
    mass0 = dust_mass(rho_d_hat, dz_hat)
    dt_hat = stable_timestep(dz_hat, u_hat, alpha)

    print("\n" + "-" * 60)
    print(f"alpha = {alpha:.0e}: {t_hat_final / dt_hat:.0f} steps")
    start = time.time()

    t_hat = 0.0
    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(rho_d_hat, z_hat, dt_hat, u_hat, alpha)
        t_hat += dt_hat

    print(f"Time taken: {time.time() - start:.1f} seconds")
    print(f"Dust mass now / initial = {dust_mass(rho_d_hat, dz_hat) / mass0:.12f}")

    epsilon = rho_d_hat / rho_g_hat
    ratio_1H = epsilon[i_1H] / epsilon[0]
    rms_height = np.sqrt(np.sum(rho_d_hat * z_hat**2) / np.sum(rho_d_hat))
    print(f"midplane dust-to-gas ratio          = {epsilon[0]:.3e}")
    print(f"r.m.s. dust height                  = {rms_height:.2f} H")
    print(f"epsilon(z = 1H) / epsilon(midplane) = {ratio_1H:.3e} "
          f"({'above' if ratio_1H > np.exp(-1) else 'below'} 1/e = 0.368)")

    label = "advection only (alpha = 0)" if alpha == 0 else f"alpha = {alpha:.0e}"
    line, = plt.plot(z_hat, epsilon, label=label)
    if alpha > 0:
        steady = steady_state_profile(z_hat, St0, alpha, z_max=z_max) / rho_g_hat
        plt.plot(z_hat, steady, ":", color=line.get_color())

# In steady state epsilon(z)/epsilon(0) = exp(-(St0/alpha)(exp(z^2/2) - 1)), so the
# dust is depleted by 1/e at z = 1H (the observed height) when
alpha_obs = St0 * (np.exp(0.5) - 1.0)
print("\n" + "-" * 60)
print(f"alpha that puts the 1/e depletion height at z = 1H: {alpha_obs:.2e}")

plt.plot([], [], ":", color="k", label="analytic steady state")
plt.yscale("log")
plt.ylim(1e-6, 1.0)
plt.xlabel("z / H")
plt.ylabel(r"$\rho_d$/$\rho_g$")
plt.title(f"1 µm dust at {R_AU:.0f} AU after 1 Myr for different alpha")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
