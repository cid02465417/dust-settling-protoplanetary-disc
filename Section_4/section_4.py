import numpy as np
import matplotlib.pyplot as plt
from dust_functions import (
    temperature_R, rho_g0_R, sound_speed, Omega_K, scale_height,
    St0_from_R_a, make_z_grid, gas_density_dimless,
    velocity_dimless, dust_step, dust_to_gas_ratio_dimless
)

# Parameters

R_AU    = 1
a_grain = 1.0e-6

# Disc properties at this radius
T_R     = temperature_R(R_AU)
cs_R    = sound_speed(R_AU)
Omega_R = Omega_K(R_AU)
H_R     = scale_height(R_AU)
rho_g0  = rho_g0_R(R_AU)

# Dimensionless midplane settling time (Stopping time)
St0 = St0_from_R_a(R_AU, a_grain)

print(f"R = {R_AU:.2f} AU")
print(f"T(R)     = {T_R:.3e} K")
print(f"c_s(R)   = {cs_R:.3e} m/s")
print(f"Omega(R) = {Omega_R:.3e} 1/s")
print(f"H(R)     = {H_R:.3e} m")
print(f"rho_g0   = {rho_g0:.3e} kg/m^3")
print(f"a_grain  = {a_grain:.3e} m")
print(f"St0      = {St0:.3e}")
print()

# grid setup

z_max_hat = 3.0
N         = 1600

z_hat, dz_hat = make_z_grid(z_max=z_max_hat, N=N)
u_hat = velocity_dimless(z_hat, St0)

# CFL timestep
cfl     = 0.5
u_max   = np.max(np.abs(u_hat))
dt_hat  = cfl * dz_hat / u_max


# Final dimensionless time: 50500000.0 ≈ 8 Myr at 1 AU, 63000000.0 ≈ 10 Myr at 1 AU
t_hat_final = 50500000.0


print(f"dz_hat       = {dz_hat:.4e}")
print(f"max|u_hat|   = {u_max:.4e}")
print(f"dt_hat (CFL) = {dt_hat:.4e}")
print(f"Estimated steps ≈ {t_hat_final / dt_hat:.0f}")
print()

# Convert to physical time
year = 365.25 * 24 * 3600.0
t_phys_final = t_hat_final / Omega_R
print(f"t_final ≈ {t_phys_final/year:.3e} years")


rho_g_hat = gas_density_dimless(z_hat)
rho_d_hat = 0.01 * rho_g_hat      # Initial dust profile: ρ_d = 0.01 ρ_g


# Time integration and checking epsilon_midplane > 1


t_hat = 0.0
n_steps = 0

# Plotting snapshots
save_times_hat = [0.0, t_hat_final/2, t_hat_final]
saved_profiles = [rho_d_hat.copy()]
saved_t_hat    = [t_hat]
next_save_idx  = 1

# Track when epsilon_max > 1
epsilon_history = []
time_when_epsilon_gt1 = None


while t_hat < t_hat_final:
    # Advance dust density
    rho_d_hat = dust_step(rho_d_hat, dz_hat, dt_hat, u_hat)
    t_hat += dt_hat
    n_steps += 1

    # Compute dust-to-gas ratio
    epsilon = dust_to_gas_ratio_dimless(rho_d_hat, z_hat)
    epsilon_max = np.max(epsilon)   # at midplane (z_hat=0)
    imax = np.argmax(epsilon)
    z_loc = z_hat[imax]
    epsilon_history.append((t_hat, epsilon_max, z_loc))

    if time_when_epsilon_gt1 is None and epsilon_max > 1.0:
        time_when_epsilon_gt1 = t_hat
        z_when_epsilon_gt1    = z_loc

    # Save profiles at selected times
    if next_save_idx < len(save_times_hat):
        target = save_times_hat[next_save_idx]
        if t_hat >= target:
            saved_profiles.append(rho_d_hat.copy())
            saved_t_hat.append(t_hat)

            next_save_idx += 1

print(f"Finished at t_hat = {t_hat:.3f} after {n_steps} steps.")

if time_when_epsilon_gt1 is not None:
    t_phys = time_when_epsilon_gt1 / Omega_R
    print(f"ε > 1 first at t_hat ≈ {time_when_epsilon_gt1:.3f}, "
          f"t ≈ {t_phys/year:.1e} years, "
          f"at z/H ≈ {z_when_epsilon_gt1:.3e} "
          f"with ε = {epsilon_history[-1][1]} at {t_phys_final/year:.3e} years")
else:
    print("ε never exceeded 1 within t_hat_final.")


# Plot snapshots of dust density


plt.figure()
for prof, tt in zip(saved_profiles, saved_t_hat):
    plt.plot(z_hat, prof, label=f"t ≈ {(tt/Omega_R/year/10**6) :.1f} Myr")
plt.xlabel(r"$z/H$")
plt.ylabel(r"${\rho}_d$")
plt.title(f"Advection-only dust settling at R = {R_AU:.1f} AU, "+r"$m_d$"+ f" = {a_grain*1e6:.1f} μm")
plt.legend()
plt.tight_layout()
plt.show()


# Plot epsilon_max vs time


if epsilon_history:
    times_hat, eps_vals, z_locs = zip(*epsilon_history)
    times_years = np.array(times_hat) / Omega_R / year

    plt.figure()
    plt.plot(times_years/10**6, eps_vals)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("time [Myr]")
    plt.ylabel(r"$\rho_d/\rho_g$")
    plt.title(f"Midplane dust-to-gas ratio for advection at R = {R_AU:.1f} AU, "+r"$m_d$"+ f" = {a_grain*1e6:.1f} μm")
    plt.tight_layout()
    plt.show()
