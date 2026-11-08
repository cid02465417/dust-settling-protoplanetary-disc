import numpy as np
import matplotlib.pyplot as plt
import time

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

# Radii studied in the report (one at a time): 5, 10, 20, 30 AU
# (roughly Jupiter, Saturn, Uranus, Neptune).

R_AU = 10.0
alpha = 1e-4

z_max = 3.0
N_z = 200


Omega_R = Omega_K(R_AU)
year = 365.25 * 24 * 3600.0  # seconds in a year


t_hat_final = 1e6 * year * Omega_R  # final dimensionless time


# Grid and gas profile

z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)

rho_g_hat = gas_density_dimless(z_hat)


# List of grain sizes to test (in metres)

# Uncomment more sizes to scan; each one adds to the run time.
a_list = [
    1e-6,     # 1 µm
    # 3e-6,   # 3 µm
    # 2e-5,   # 20 µm
    # 3e-5,   # 30 µm
    # 5e-5,   # 50 µm
    # 6e-5,   # 60 µm
    # 1e-4,   # 100 µm
    # 1.5e-4, # 150 µm
    # 1.9e-4, # 190 µm
    # 1e-3,   # 1 mm
    # 3e-3,   # 3 mm
    # 1e-2,   # 1 cm
    # 2e-2,   # 2 cm
    # 3e-2,   # 3 cm
]

# storage for results
results = []   # (a_m, time_eps_gt1_hat, time_eps_gt1_yr, final_rho_d_hat)


start_all = time.time()

for a_m in a_list:
    print("\n" + "-"*60)
    print(f"Grain size a = {a_m:.2e} m ({a_m*1e6:.1f} µm)")


    # Initial dust, stopping time, velocity

    rho_d_hat = 0.01 * rho_g_hat.copy()
    St0 = St0_from_R_a(R_AU, a_m)
    u_hat = velocity_dimless(z_hat, St0)


    # dt_hat from advection and diffusion CFL

    umax = np.max(np.abs(u_hat))
    dt_adv = 0.4 * dz_hat / umax
    dt_diff = max_dt_hat_for_diffusion(dz_hat, alpha, cfl=0.4)
    dt_hat = min(dt_adv, dt_diff)       # choose the smaller one for a safe step

    est_steps = t_hat_final / dt_hat
    print(f"Estimated steps ≈ {est_steps:.0f}")

    # convert to physical time
    t_phys_final = t_hat_final / Omega_R
    print(f"t_final ≈ {t_phys_final/year:.3e} years")


    # Time evolution

    t_hat = 0.0
    time_eps_gt1_hat = None

    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(
            rho_d_hat, z_hat, dt_hat, u_hat, alpha
        )
        t_hat += dt_hat

        # track midplane epsilon
        epsilon_mid = rho_d_hat[0] / rho_g_hat[0]

        if (time_eps_gt1_hat is None) and (epsilon_mid > 1.0):
            time_eps_gt1_hat = t_hat
            t_phys = t_hat / Omega_R
            print(f" --> epsilon_mid > 1 reached at "
                  f"t_hat = {t_hat:.2e}, t ≈ {t_phys/year:.3e} years")


    # store results
    if time_eps_gt1_hat is None:
        t_eps_yr = None
        print(" --> epsilon_mid <= 1 for entire run.")
    else:
        t_eps_yr = (time_eps_gt1_hat / Omega_R) / year

    results.append((a_m, time_eps_gt1_hat, t_eps_yr, rho_d_hat.copy()))

end_all = time.time()
print("\nTotal wall-clock time:", end_all - start_all, "seconds")


# Plot final epsilon(z) profiles for all sizes

plt.figure(figsize=(6,5))
to_plot = [[], []]

for a_m, t_hat_eps, t_eps_yr, rho_d_final in results:
    epsilon_final = rho_d_final / rho_g_hat
    to_plot[0].append(a_m)
    to_plot[1].append(epsilon_final[0])


to_plot[0] = np.array(to_plot[0])*1e6  # convert to microns
plt.plot(to_plot[0], to_plot[1], '--o')
plt.title(f"Dust-to-gas ratio for varying dust particle sizes at R = {R_AU:.1f} AU, alpha = {alpha:.1e}")
plt.xlabel("dust grain size (µm)")
plt.ylabel("epsilon = rho_d / rho_g")
plt.grid(True)

plt.tight_layout()
plt.show()


print(f"\nSummary (R = {R_AU:.0f} AU, alpha = {alpha:.1e}):")
print("a_m [m]   a [µm]   t(eps>1) [years] (None = never reached)")
for a_m, t_hat_eps, t_eps_yr, _ in results:
    if t_eps_yr is None:
        print(f"{a_m:8.2e}  {a_m*1e6:7.1f}   {None}")
    else:
        print(f"{a_m:8.2e}  {a_m*1e6:7.1f}   {t_eps_yr: .3e}")


# Thresholds found by running this script at each radius (values from the report)
plt.figure()
plt.plot([5.0, 10.0, 20.0, 30.0], [50, 70, 120, 200], marker="o")
plt.xlabel("Radius R (AU)")
plt.ylabel("Minimum grain size for ε > 1 at 1 Myr (μm)")
plt.title("Planetesimal formation threshold (ε > 1) after 1 Myr")
plt.grid(True)
plt.tight_layout()
plt.show()
