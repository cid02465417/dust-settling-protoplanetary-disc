"""
Step IC demo: accuracy vs analytic solution, convergence,
numerical diffusion and CFL instability of the upwind scheme.
"""


import numpy as np
import matplotlib.pyplot as plt
from functions import upwind_step, run_advection_once


def step_IC(x, x_left=-1.0, x_right=1.0):
    u0 = np.zeros_like(x)
    mask = (x >= x_left) & (x <= x_right)
    u0[mask] = 1.0
    return u0

def analytic_solution_step(x, t, v, x_min, x_max,
                           x_left=-1.0, x_right=1.0):
    Lx = x_max - x_min
    x_back = x - v * t
    x_wrapped = ((x_back - x_min) % Lx) + x_min
    return step_IC(x_wrapped, x_left=x_left, x_right=x_right)


# Parameters

x_min, x_max = -5.0, 5.0
N = 200
v = 1.0
T_final = 2.0
cfl = 0.5
x_left, x_right = -1.0, 1.0


# Run advection

x, u_num, dx, t = run_advection_once(
    N=N,
    T_final=T_final,
    step_func=upwind_step,
    init_func=lambda x: step_IC(x, x_left=x_left, x_right=x_right),
    v=v,
    x_min=x_min,
    x_max=x_max,
    cfl=cfl
)

# Analytic solution
u_exact = analytic_solution_step(
    x, t, v, x_min, x_max,
    x_left=x_left, x_right=x_right
)

# Initial condition
u0 = step_IC(x, x_left=x_left, x_right=x_right)

# Error
error_L2 = np.sqrt(np.mean((u_num - u_exact)**2))
norm_exact = np.sqrt(np.mean(u_exact**2))
rel_error_L2 = error_L2 / norm_exact
print(f"N = {N}, dx = {dx:.3e}, rel L2 error = {rel_error_L2:.3e}")


plt.figure()
plt.plot(x, u0, label="initial")
plt.plot(x, u_exact, "--", label="analytic (shifted step)")
plt.plot(x, u_num, label="numerical")
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Upwind advection of step at t = {t:.1f}")
plt.legend()
plt.show()


Ns = [20, 40, 80, 160, 320, 640]
x_min, x_max = -5.0, 5.0
T_final = 2.0
v = 1.0
cfl = 0.5
x_left, x_right = -1.0, 1.0

dx_list = []
err_list = []

print("\nConvergence test (step IC):")
for N_test in Ns:
    x, u_num, dx, t = run_advection_once(
        N=N_test,
        T_final=T_final,
        step_func=upwind_step,
        init_func=lambda x: step_IC(x, x_left=x_left, x_right=x_right),
        v=v,
        x_min=x_min,
        x_max=x_max,
        cfl=cfl
    )

    u_exact = analytic_solution_step(
        x, t, v, x_min, x_max,
        x_left=x_left, x_right=x_right
    )

    error_L2 = np.sqrt(np.mean((u_num - u_exact)**2))
    norm_exact = np.sqrt(np.mean(u_exact**2))
    rel_error_L2 = error_L2 / (norm_exact + 1e-14)

    dx_list.append(dx)
    err_list.append(rel_error_L2)

    print(f"N = {N_test:4d}, dx = {dx:.4e}, rel L2 err = {rel_error_L2:.4e}")


dx_arr = np.array(dx_list)
err_arr = np.array(err_list)

plt.figure()
plt.loglog(dx_arr, err_arr, "o-", label="upwind (step IC)")
plt.gca().invert_xaxis()
plt.xlabel(r"$\Delta x$")
plt.ylabel("relative L2 error")
plt.title("Convergence of upwind scheme (step initial condition)")
plt.legend()
plt.show()

# Estimate slope
slope = np.polyfit(np.log(dx_arr), np.log(err_arr), 1)[0]
print(f"Estimated convergence order (slope) ≈ {slope:.2f}")


# Numerical diffusion demo (step IC)


# Parameters for demo
N_demo = 400
x_min_demo, x_max_demo = -10.0, 10.0
Lx_demo = x_max_demo - x_min_demo
dx_demo = Lx_demo / N_demo
x_demo = np.linspace(x_min_demo, x_max_demo - dx_demo, N_demo)

v_demo = 1.0
cfl_demo = 0.5
dt_demo = cfl_demo * dx_demo / abs(v_demo)

T_final_demo = 8.0   # long enough to see smearing
x_left_demo, x_right_demo = -1.0, 1.0

# Initial condition: sharp step
u_demo = step_IC(x_demo, x_left=x_left_demo, x_right=x_right_demo)

# Times at which to save profiles
times_to_save = [0.0,
                 T_final_demo / 4,
                 T_final_demo / 2,
                 3 * T_final_demo / 4,
                 T_final_demo]

saved_profiles = []
saved_times = []

t_demo = 0.0

# Save initial profile
saved_profiles.append(u_demo.copy())
saved_times.append(t_demo)

# Time stepping
while t_demo < T_final_demo:
    u_demo = upwind_step(u_demo, dx_demo, dt_demo, v=v_demo)
    t_demo += dt_demo

    # Save when we cross one of the target times
    if len(saved_times) < len(times_to_save):
        next_target = times_to_save[len(saved_times)]
        if t_demo >= next_target:
            saved_profiles.append(u_demo.copy())
            saved_times.append(t_demo)

# Plot the profiles
plt.figure()
for u_prof, tt in zip(saved_profiles, saved_times):
    plt.plot(x_demo, u_prof, label=f"t ≈ {tt:.2f}")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Numerical diffusion of step (upwind scheme)")
plt.legend()
plt.show()


# Unstable CFL demo


N_unstable = 400
x_min_u, x_max_u = -5.0, 5.0
Lx_u = x_max_u - x_min_u
dx_u = Lx_u / N_unstable
x_u = np.linspace(x_min_u, x_max_u - dx_u, N_unstable)

v_unstable = 1.0
cfl_unstable = 1.01          # deliberately violate CFL <= 1
dt_u = cfl_unstable * dx_u / abs(v_unstable)

T_final_unstable = 2.0
x_left_u, x_right_u = -1.0, 1.0

u_u = step_IC(x_u, x_left=x_left_u, x_right=x_right_u)

t_u = 0.0
u_history = []
t_history = []


while t_u < T_final_unstable:
    u_history.append(u_u.copy())
    t_history.append(t_u)

    u_u = upwind_step(u_u, dx_u, dt_u, v=v_unstable)
    t_u += dt_u

# Plot initial vs final to show instability
plt.figure()
plt.plot(x_u, step_IC(x_u, x_left=x_left_u, x_right=x_right_u),
         label="initial")
plt.plot(x_u, u_u, label=f"final (CFL = {cfl_unstable})")
plt.xlabel("x")
plt.ylabel("u")
plt.title("Unstable solution with CFL > 1 (step IC)")
plt.legend()
plt.show()
