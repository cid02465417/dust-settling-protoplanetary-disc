"""
Gaussian IC demo: accuracy vs analytic solution, convergence,
numerical diffusion and CFL instability of the upwind scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import upwind_step, run_advection_once, analytic_solution_gaussian


def gaussian_IC(x, L_gauss=1.0):
    """
    Gaussian initial condition: u0(x) = exp(-(x^2)/L^2)
    """
    return np.exp(-(x**2) / L_gauss**2)


# Parameters

L_gauss = 1.0          # L in exp(-x^2 / L^2)
x_min, x_max = -5.0, 5.0   # domain
N = 200           # number of grid points
v = 1.0                # advection velocity (>0 for now)

T_final = 2.0


# CFL choice (must be <= 1 for stability)
cfl = 0.5


# Run advection


x, u_num, dx, t = run_advection_once(
    N=N,
    T_final=T_final,
    step_func=upwind_step,
    init_func=lambda x: gaussian_IC(x, L_gauss),
    v=v,
    x_min=x_min,
    x_max=x_max,
    cfl=cfl
)


# Analytic comparison


u_exact = analytic_solution_gaussian(x, t, v, L_gauss, x_min, x_max)

error_L2 = np.sqrt(np.mean((u_num - u_exact)**2))
norm_exact = np.sqrt(np.mean(u_exact**2))
rel_error_L2 = error_L2 / norm_exact

print(f"N = {N}")
print(f"dx = {dx:.4e}")
print(f"final time = {t:.4f}")
print(f"L2 error = {error_L2:.3e}")
print(f"Relative L2 error = {rel_error_L2:.3e}")


# Quick plot


u0 = gaussian_IC(x, L_gauss=L_gauss) #initial condition

plt.figure()
plt.plot(x, u0, label="initial")
plt.plot(x, u_exact, "--", label="analytic")
plt.plot(x, u_num, label="numerical")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Upwind advection of Gaussian at t = {t:.1f}")
plt.show()


# Convergence loop


Ns = [
    50, 80, 128,
    200, 320, 512,
    800, 1000, 1500,
    2000, 3000, 4000
]

dx_list = []
err_list = []

print("\nConvergence test:")
for N_test in Ns:
    x, u_num, dx, t = run_advection_once(
        N=N_test,
        T_final=T_final,
        step_func=upwind_step,
        init_func=lambda x: gaussian_IC(x, L_gauss),
        v=v,
        x_min=x_min,
        x_max=x_max,
        cfl=cfl
    )

    u_exact = analytic_solution_gaussian(x, t, v, L_gauss, x_min, x_max)
    error_L2 = np.sqrt(np.mean((u_num - u_exact)**2))
    norm_exact = np.sqrt(np.mean(u_exact**2))
    rel_error_L2 = error_L2 / norm_exact

    dx_list.append(dx)
    err_list.append(rel_error_L2)

    print(f"N = {N_test:4d}, dx = {dx:.4e}, rel L2 err = {rel_error_L2:.4e}")

# Log-log convergence plot
dx_arr = np.array(dx_list)
err_arr = np.array(err_list)

plt.figure()
plt.loglog(dx_arr, err_arr, "o-", label="upwind scheme")
plt.gca().invert_xaxis()
plt.xlabel(r"$\Delta x$")
plt.ylabel("relative L2 error")
plt.title("Convergence of upwind advection")
plt.legend()
plt.show()


slope = np.polyfit(np.log(dx_arr), np.log(err_arr), 1)[0]
print(f"\nEstimated convergence order ≈ {slope:.2f}")


# Numerical diffusion demo


# Use a moderate resolution
N_demo = 200
T_final_demo = 8.0
v_demo = 1.0
x_min, x_max = -10.0, 10.0
L_gauss_demo = 1.0
cfl_demo = 0.5

# Set up grid
Lx_demo = x_max - x_min
dx_demo = Lx_demo / N_demo
x_demo = np.linspace(x_min, x_max - dx_demo, N_demo)

# Initial condition
u_demo = gaussian_IC(x_demo, L_gauss=L_gauss_demo)

# Time step from CFL
dt_demo = cfl_demo * dx_demo / abs(v_demo)

# Times at which to save profiles
times_to_save = [0.0, T_final_demo / 4, T_final_demo / 2, 3 * T_final_demo / 4,  T_final_demo]
saved_profiles = []
saved_times = []

t_demo = 0.0

# Save initial profile
saved_profiles.append(u_demo.copy())
saved_times.append(t_demo)

while t_demo < T_final_demo:
    # Advance one step
    u_demo = upwind_step(u_demo, dx_demo, dt_demo, v=v_demo)
    t_demo += dt_demo

    if saved_times[-1] < times_to_save[len(saved_times)] <= t_demo and len(saved_times) < len(times_to_save):
        saved_profiles.append(u_demo.copy())
        saved_times.append(t_demo)

# Plot profiles
plt.figure()
for u_prof, tt in zip(saved_profiles, saved_times):
    plt.plot(x_demo, u_prof, label=f"t ≈ {tt:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Numerical diffusion of Gaussian (upwind scheme)")
plt.legend()
plt.show()


# Unstable CFL demo (CFL > 1)


N_unstable = 200
T_final_unstable = 2.0
v_unstable = 1.0
x_min, x_max = -5.0, 5.0
L_gauss_unstable = 1.0

# Set up grid
Lx_unstable = x_max - x_min
dx_unstable = Lx_unstable / N_unstable
x_unstable = np.linspace(x_min, x_max - dx_unstable, N_unstable)

# Initial condition
u_unstable = gaussian_IC(x_unstable, L_gauss=L_gauss_unstable)

# Choose CFL > 1 on purpose
cfl_unstable = 4
dt_unstable = cfl_unstable * dx_unstable / abs(v_unstable)

t_unstable = 0.0
u_history = []
t_history = []

while t_unstable < T_final_unstable:
    u_history.append(u_unstable.copy())
    t_history.append(t_unstable)

    u_unstable = upwind_step(u_unstable, dx_unstable, dt_unstable, v=v_unstable)
    t_unstable += dt_unstable

# Plot initial vs final snapshot to show the blow-up
plt.figure()
plt.plot(x_unstable, gaussian_IC(x_unstable, L_gauss=L_gauss_unstable),
         label="initial")
plt.plot(x_unstable, u_unstable, label=f"final (CFL = {cfl_unstable})")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Unstable solution with CFL > 1")
plt.legend()
plt.show()
