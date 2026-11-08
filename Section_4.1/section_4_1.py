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

R_AU = 100.0
a_m  = 1e-6
alphas = [0, 1e-5, 1e-4, 1e-3]   # turbulence strengths to compare

z_max = 3.0
N_z = 400
t_hat_final = 10000.0   # time approx 1Myr at 100 AU

plt.figure(figsize=(6,5))

counter_zero = 0
for alpha in alphas:
    Omega_R = Omega_K(R_AU)

    # Grid and initial conditions

    z_hat, dz_hat = make_z_grid(z_max=z_max, N=N_z)

    # Gas + initial dust

    rho_g_hat = gas_density_dimless(z_hat)
    rho_d_hat = 0.01 * rho_g_hat.copy()

    # Stokes number and velocity field
    St0 = St0_from_R_a(R_AU, a_m)
    u_hat = velocity_dimless(z_hat, St0)


    # Choose dt_hat based on advection + diffusion CFL

    umax = np.max(np.abs(u_hat))
    dt_adv = 0.4 * dz_hat / umax
    dt_diff = max_dt_hat_for_diffusion(dz_hat, alpha, cfl=0.4)
    dt_hat = min(dt_adv, dt_diff)


    # Time evolution

    t_hat = 0.0

    estimation = t_hat_final / dt_hat
    print(f"Estimated steps ≈ {estimation:.0f}")


    start = time.time()

    # Convert to physical time
    year = 365.25 * 24 * 3600.0
    t_phys_final = t_hat_final / Omega_R
    print(f"t_final ≈ {t_phys_final/year:.3e} years")


    counter = 0

    saved_profiles = [rho_d_hat.copy()]

    while t_hat < t_hat_final:
        rho_d_hat = dust_step_with_diffusion(
            rho_d_hat, z_hat, dt_hat, u_hat, alpha
        )
        if counter == 0 and t_hat > t_hat_final/3:
            saved_profiles.append(rho_d_hat.copy())
            counter += 1
        elif counter == 1 and t_hat > 2*t_hat_final/3:
            saved_profiles.append(rho_d_hat.copy())
            counter += 1
        t_hat += dt_hat

    saved_profiles.append(rho_d_hat.copy())

    end = time.time()

    print(f"Time taken: {(end - start):.3} seconds")

    counter = 0
    for i in z_hat:
        if i >= 1.0:
            index_at_z_1 = z_hat[counter]
            break
        counter += 1

    if rho_d_hat[counter]/rho_d_hat[0] > 0.3679:
        print(f"rho_d_hat at z=1H: {rho_d_hat[counter]/rho_d_hat[0]:.3e} and is bigger than 1/e = 0.3679")
    else:
        print(f"rho_d_hat at z=1H: {rho_d_hat[counter]/rho_d_hat[0]:.3e} and is smaller than 1/e = 0.3679")


    print(f"For alpha = {alpha:.1e} the final midplane dust-to-gas ratio epsilon_mid = {rho_d_hat[0]/rho_g_hat[0]:.3e}")


    # Plot final profile

    if counter_zero == 0:
        plt.plot(z_hat, saved_profiles[2]/rho_g_hat, label="Advection only (alpha=0)")
        counter_zero += 1
        continue
    plt.plot(z_hat, saved_profiles[2]/rho_g_hat, label=f"alpha = {alpha:.0e} (t_final)")
    plt.xlabel("z / H")
    plt.ylabel(r"$\rho_d$/$\rho_g$")
    plt.title("Dust profile with turbulence for different alpha values")
    plt.legend()
    plt.grid(True)

plt.show()
