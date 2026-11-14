"""
Section 5: how large must grains grow to reach epsilon > 1 against turbulence?

With turbulence the dust layer reaches a steady state in which settling is
balanced by diffusion. Its midplane dust-to-gas ratio depends only on
St0 / alpha, and has an analytic form (diffusion_functions.steady_state_profile).

Part 1 runs the solver at one radius for a few grain sizes and compares the
final midplane epsilon with the analytic steady state.
Part 2 uses the analytic steady state to find the threshold grain size for
epsilon > 1 at the radii of the giant planets.

The thin layers involved (H_dust ~ 0.01 H at the threshold) need far more grid
points than is practical for the explicit solver, which is why the threshold
itself is taken from the analytic result that Part 1 validates.
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
    response_time,
    steady_state_midplane_epsilon,
    threshold_grain_size,
)

# Parameters

alpha = 1e-4
z_max = 3.0

# Part 1: numerical runs at one radius
R_AU = 10.0
N_z = 400
t_max_years = 1.0e6          # never run longer than the disc lifetime

# Grain sizes to simulate (metres). Each takes roughly a minute.
a_list = [
    1e-6,     # 1 µm
    1e-4,     # 100 µm
    1e-3,     # 1 mm
    # 1e-2,   # 1 cm  (dust layer only a few cells thick at N_z = 400)
]

# Part 2: radii of the giant planets (AU)
radii = [5.0, 10.0, 20.0, 30.0]


# ------------------------------------------------------------
# Part 1: solver against the analytic steady state
# ------------------------------------------------------------

Omega_R = Omega_K(R_AU)
z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)
rho_g_hat = gas_density_dimless(z_hat)

results = []   # (a, St0, eps_numerical, eps_analytic, years_run)
start_all = time.time()

for a_m in a_list:
    St0 = St0_from_R_a(R_AU, a_m)
    u_hat = face_velocity(z_hat, dz_hat, St0)
    dt_hat = stable_timestep(dz_hat, u_hat, alpha)

    # run for 15 response times (steady state), capped at the disc lifetime
    t_hat_final = min(15.0 * response_time(St0, alpha), t_max_years * YEAR * Omega_R)

    print("\n" + "-" * 60)
    print(f"a = {a_m:.1e} m, St0 = {St0:.2e}, St0/alpha = {St0/alpha:.2e}, "
          f"{t_hat_final / dt_hat:.0f} steps")

    rho_d_hat = 0.01 * rho_g_hat
    mass0 = dust_mass(rho_d_hat, dz_hat)
    t_hat = 0.0
    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(rho_d_hat, z_hat, dt_hat, u_hat, alpha)
        t_hat += dt_hat

    eps_num = rho_d_hat[0] / rho_g_hat[0]
    eps_ana = steady_state_midplane_epsilon(St0, alpha, z_max=z_max)
    years = t_hat / Omega_R / YEAR
    results.append((a_m, St0, eps_num, eps_ana, years))
    print(f"ran for {years:.2e} years; mass now / initial = "
          f"{dust_mass(rho_d_hat, dz_hat) / mass0:.12f}")
    print(f"midplane epsilon: numerical {eps_num:.4f}, analytic steady state {eps_ana:.4f}")

print(f"\nTotal wall-clock time: {time.time() - start_all:.0f} seconds")

a_line = np.logspace(-6, 0, 200)
eps_line = [steady_state_midplane_epsilon(St0_from_R_a(R_AU, a), alpha, z_max=z_max)
            for a in a_line]

plt.figure(figsize=(6.5, 5))
plt.loglog(a_line * 1e6, eps_line, label="analytic steady state")
plt.loglog([r[0] * 1e6 for r in results], [r[2] for r in results], "o",
           label=f"solver (N = {N_z})")
plt.axhline(1.0, linestyle=":", color="k")
plt.xlabel("dust grain size (µm)")
plt.ylabel("midplane epsilon = rho_d / rho_g")
plt.title(f"Steady-state midplane dust-to-gas ratio at R = {R_AU:.0f} AU, alpha = {alpha:.0e}")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Part 2: threshold grain size at the giant-planet radii
# ------------------------------------------------------------

print(f"\nGrain size needed for a steady-state midplane epsilon > 1 (alpha = {alpha:.0e})")
print("R [AU]   a_threshold     St0 at threshold   time to settle [yr]")
a_thr = []
for R in radii:
    a_c = threshold_grain_size(R, alpha, z_max=z_max)
    St0_c = St0_from_R_a(R, a_c)
    t_settle = 10.0 * response_time(St0_c, alpha) / Omega_K(R) / YEAR
    a_thr.append(a_c)
    print(f"{R:6.1f}   {a_c*100:8.2f} cm     {St0_c:10.2f}         {t_settle:10.1f}")

print("\nSt0 at the threshold is of order 1, where the terminal-velocity "
      "approximation (valid for St << 1) no longer holds.")

plt.figure()
plt.loglog(radii, np.array(a_thr) * 100, marker="o")
plt.xlabel("Radius R (AU)")
plt.ylabel("Minimum grain size for ε > 1 (cm)")
plt.title(f"Planetesimal formation threshold (ε > 1), alpha = {alpha:.0e}")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
