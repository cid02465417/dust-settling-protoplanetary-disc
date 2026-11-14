"""
Section 4: advection-only settling of 1 micron dust.

Part 1 evolves the dust at one radius and records when the midplane
dust-to-gas ratio epsilon first exceeds 1.
Part 2 repeats the measurement at several radii and compares it with the
exact result  t(epsilon = 1) = ln(100) / (St0 * Omega).
"""

import numpy as np
import matplotlib.pyplot as plt
from dust_functions import (
    temperature_R, rho_g0_R, sound_speed, Omega_K, scale_height,
    St0_from_R_a, make_z_grid, gas_density_dimless,
    face_velocity, dust_step, advection_timestep, dust_mass,
    midplane_epsilon_advection, time_to_epsilon_one_advection, YEAR,
)

# Parameters

R_AU    = 1.0
a_grain = 1.0e-6
z_max_hat = 3.0
N         = 800
cfl       = 0.5
t_final_years = 10.0e6       # 10 Myr; the disc itself only lasts ~1 Myr

Myr = 1.0e6 * YEAR


def time_to_epsilon_one(R, a, N, z_max=3.0, cfl=0.5, t_max_hat=None,
                        record=False):
    """
    Evolve advection-only settling at radius R (AU) for grain size a (m).

    Returns the dimensionless time at which the midplane epsilon first
    exceeds 1 (None if it never does before t_max_hat), plus the recorded
    history and snapshots if record is True.
    """
    St0 = St0_from_R_a(R, a)
    z, dz = make_z_grid(z_max=z_max, N=N)
    rho_g = gas_density_dimless(z)
    u = face_velocity(z, dz, St0)
    dt = advection_timestep(dz, u, cfl=cfl)
    if t_max_hat is None:
        t_max_hat = 1.5 * time_to_epsilon_one_advection(St0)

    rho = 0.01 * rho_g
    mass0 = dust_mass(rho, dz)
    t, t_cross = 0.0, None
    times, eps_mid, snapshots = [0.0], [0.01], [(0.0, rho.copy())]
    snap_times = [t_max_hat / 2, t_max_hat]

    while t < t_max_hat:
        rho = dust_step(rho, dz, dt, u)
        t += dt
        eps = rho[0] / rho_g[0]
        if t_cross is None and eps > 1.0:
            t_cross = t
            if not record:
                break
        if record:
            times.append(t)
            eps_mid.append(eps)
            if snap_times and t >= snap_times[0]:
                snapshots.append((t, rho.copy()))
                snap_times.pop(0)

    mass_ratio = dust_mass(rho, dz) / mass0
    return t_cross, z, np.array(times), np.array(eps_mid), snapshots, mass_ratio


# ------------------------------------------------------------
# Part 1: one radius
# ------------------------------------------------------------

Omega_R = Omega_K(R_AU)
St0 = St0_from_R_a(R_AU, a_grain)

print(f"R = {R_AU:.2f} AU")
print(f"T(R)     = {temperature_R(R_AU):.3e} K")
print(f"c_s(R)   = {sound_speed(R_AU):.3e} m/s")
print(f"Omega(R) = {Omega_R:.3e} 1/s")
print(f"H(R)     = {scale_height(R_AU):.3e} m")
print(f"rho_g0   = {rho_g0_R(R_AU):.3e} kg/m^3")
print(f"a_grain  = {a_grain:.3e} m")
print(f"St0      = {St0:.3e}")
print()

t_hat_final = t_final_years * YEAR * Omega_R
t_cross, z_hat, times_hat, eps_vals, snapshots, mass_ratio = time_to_epsilon_one(
    R_AU, a_grain, N, z_max=z_max_hat, cfl=cfl, t_max_hat=t_hat_final, record=True
)

t_exact = time_to_epsilon_one_advection(St0)
print(f"Dust mass at the end / initial = {mass_ratio:.12f}")
if t_cross is not None:
    print(f"ε > 1 first at t ≈ {t_cross / Omega_R / Myr:.2f} Myr (numerical, N = {N})")
else:
    print("ε never exceeded 1 within the run.")
print(f"Exact result ln(100)/St0      = {t_exact / Omega_R / Myr:.2f} Myr")

# Snapshots of dust density
plt.figure()
for tt, prof in snapshots:
    plt.plot(z_hat, prof, label=f"t ≈ {tt / Omega_R / Myr:.1f} Myr")
plt.yscale("log")
plt.ylim(1e-6, 10)
plt.xlabel(r"$z/H$")
plt.ylabel(r"$\rho_d \,/\, \rho_g(R,0)$")
plt.title(f"Advection-only dust settling at R = {R_AU:.1f} AU, a = {a_grain*1e6:.1f} μm")
plt.legend()
plt.tight_layout()
plt.show()

# Midplane epsilon against time, with the exact solution
times_Myr = times_hat / Omega_R / Myr
plt.figure()
plt.plot(times_Myr, eps_vals, label=f"numerical (N = {N})")
plt.plot(times_Myr, midplane_epsilon_advection(times_hat, St0), "--",
         label=r"exact: $0.01\,e^{\mathrm{St}_0 \hat t}$")
plt.axhline(1.0, linestyle=":", color="k")
plt.yscale("log")
plt.xlabel("time [Myr]")
plt.ylabel(r"midplane $\rho_d/\rho_g$")
plt.title(f"Midplane dust-to-gas ratio, advection only, R = {R_AU:.1f} AU")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Part 2: time to reach epsilon = 1 as a function of radius
# ------------------------------------------------------------

radii = [1.0, 5.0, 10.0, 30.0, 100.0]
N_scan = 800   # the settled layer is only a few cells thick by the time ε = 1

print("\nTime for the midplane to reach ε = 1 (1 μm grains, no turbulence)")
print("R [AU]   numerical [Myr]   exact [Myr]")
t_num_Myr = []
for R in radii:
    t_c = time_to_epsilon_one(R, a_grain, N_scan, z_max=z_max_hat, cfl=cfl)[0]
    t_num_Myr.append(t_c / Omega_K(R) / Myr)
    t_ex = time_to_epsilon_one_advection(St0_from_R_a(R, a_grain)) / Omega_K(R) / Myr
    print(f"{R:6.1f}   {t_num_Myr[-1]:15.2f}   {t_ex:11.2f}")

R_line = np.logspace(0, 2, 100)
t_line = np.array([time_to_epsilon_one_advection(St0_from_R_a(R, a_grain))
                   / Omega_K(R) / Myr for R in R_line])

plt.figure()
plt.loglog(R_line, t_line, label=r"exact: $\ln(100)/(\mathrm{St}_0\,\Omega) \propto R^{-1/4}$")
plt.loglog(radii, t_num_Myr, "o", label=f"numerical (N = {N_scan})")
plt.axhline(1.0, linestyle=":", color="k", label="disc lifetime, 1 Myr")
plt.xlabel("Radius R (AU)")
plt.ylabel("time to reach ε = 1 at the midplane [Myr]")
plt.title("Settling timescale of 1 μm dust without turbulence")
plt.legend()
plt.tight_layout()
plt.show()
