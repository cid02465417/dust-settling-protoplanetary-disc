import numpy as np
import matplotlib.pyplot as plt

from dust_functions import (
    temperature_R, rho_g0_R, sound_speed, Omega_K, scale_height,
    St0_from_R_a, make_z_grid, gas_density_dimless,
    velocity_dimless, dust_step, dust_to_gas_ratio_dimless
)


# Parameters


# Disc radius where we study settling
R_AU = 1.0


a_grain = 1.0e-6    # 1 μm particles

# Compute disc properties at this radius
T_R   = temperature_R(R_AU)
cs_R  = sound_speed(R_AU)
Omega = Omega_K(R_AU)
H_R   = scale_height(R_AU)
rho_g0   = rho_g0_R(R_AU)

# Stokes number at the midplane (dimensionless)
St0 = St0_from_R_a(R_AU, a_grain)

print("Disc & grain properties at R = {:.2f} AU".format(R_AU))
print(f"T(R)           = {T_R:.3e} K")
print(f"c_s(R)         = {cs_R:.3e} m/s")
print(f"Ω(R)           = {Omega:.3e} 1/s")
print(f"H(R)           = {H_R:.3e} m")
print(f"ρ_g(R,0)       = {rho_g0:.3e} kg/m^3")
print(f"Grain size a   = {a_grain:.3e} m")
print(f"St0 (midplane) = {St0:.3e}")


# Dimensionless grid / domain

z_max_hat = 3.0
N         = 400      # number of grid points

z_hat, dz_hat = make_z_grid(z_max=z_max_hat, N=N)

# Dimensionless settling velocity u_hat(z_hat)
u_hat = velocity_dimless(z_hat, St0=St0)

# CFL number for stability (advection)
cfl = 0.5
u_max = np.max(np.abs(u_hat))
dt_hat = cfl * dz_hat / u_max   # dimensionless time step (in units of 1/Ω)

# Final dimensionless time (t_hat = Ω t)
t_hat_final = 30000000.0

print(f"dz_hat       = {dz_hat:.4e}")
print(f"max|u_hat|   = {u_max:.4e}")
print(f"dt_hat (CFL) = {dt_hat:.4e}")
print(f"Estimated number of steps ≈ {t_hat_final / dt_hat:.0f}")

# Convert dimensionless final time to physical units
t_phys_final = t_hat_final / Omega          # seconds
year = 365.25 * 24 * 3600.0
Myr  = 1.0e6 * year

print(f"t_final ≈ {t_phys_final/year:.3e} years "
      f"(≈ {t_phys_final/Myr:.3e} Myr)")


rho_g_hat = gas_density_dimless(z_hat)


rho_d = 0.01 * rho_g_hat


# Time integration


t_hat = 0.0

saved_profiles = []
saved_t_hat    = []

# save initial condition
saved_profiles.append(rho_d.copy())
saved_t_hat.append(t_hat)

# Times (in t_hat) at which to save profiles
save_times_hat = [0.0, 1000000, 5000000, 10000000, 20000000, 30000000]

next_save_index = 1  # next index in save_times_hat to hit
n_steps = 0

while t_hat < t_hat_final:
    # One dust-settling step (advection only)
    rho_d = dust_step(rho_d, dz_hat, dt_hat, u_hat)
    t_hat += dt_hat
    n_steps += 1

    # save a snapshot at save_times_hat
    if next_save_index < len(save_times_hat):
        target_time = save_times_hat[next_save_index]
        if t_hat >= target_time:
            saved_profiles.append(rho_d.copy())
            saved_t_hat.append(t_hat)
            next_save_index += 1

print(f"Finished at t_hat = {t_hat:.3f} after {n_steps} steps.")
print(f"Physical time ≈ {t_hat/Omega/year:.3e} years "
      f"(≈ {t_hat/Omega/Myr:.3e} Myr)")


# Dust-to-gas ratio


epsilon_final_midplane = dust_to_gas_ratio_dimless(rho_d, z_hat)[0]
print(f"Final ε_midplane = ρ_d/ρ_g at z=0 ≈ {epsilon_final_midplane:.3e}")


# Plot dust density profiles


plt.figure()
for prof, tt_hat in zip(saved_profiles, saved_t_hat):
    plt.plot(z_hat, prof, label=f"$\\hat t$ ≈ {tt_hat:.1f}")
plt.xlabel(r"$z/H$")
plt.ylabel(r"dust density $\hat\rho_d$ (∝ ρ_d / ρ_g(R,0))")
plt.title(f"Dust settling at R = {R_AU:.1f} AU (advection only)")
plt.legend()
plt.tight_layout()
plt.show()
