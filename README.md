# Dust Settling in Protoplanetary Discs

Year 3 computing project: numerical simulation of the vertical settling of dust
grains in a protoplanetary disc, using a first-order upwind advection scheme with
optional turbulent diffusion.

Each folder corresponds to one section of the report. The folders are meant to run
on their own, so shared modules (`dust_functions.py`, `diffusion_functions.py`) are
duplicated across them. Run the main script in a folder from inside that folder
to reproduce that section's plots and data.

> Note: docstrings and some comments were generated with Copilot or ChatGPT.

## Requirements

Python 3 with NumPy and Matplotlib:

```bash
pip install -r requirements.txt
```

## Sections

### Section 2: Upwind scheme tests

```bash
cd Section_2
python gaussian.py     # numerical diffusion, convergence and CFL instability for a Gaussian profile
python step_demo.py    # the same tests for a step function
```

### Section 3: Advection-only dust settling

```bash
cd Section_3
python dust_settling.py
```

Evolution of the dust density over time with an advection-only scheme.

### Section 4: Midplane dust-to-gas ratio

```bash
cd Section_4
python section_4.py
```

Evolves 1 µm dust at 1 AU for 10 Myr and compares the midplane dust-to-gas ratio
with the exact result `0.01 * exp(St0 * t)`. It then measures the time to reach
`epsilon = 1` at several radii and compares it with `ln(100) / (St0 * Omega)`.
Takes about 15 seconds.

### Section 4.1: Estimating alpha

```bash
cd Section_4.1
python section_4_1.py
```

Compares dust profiles at 100 AU after 1 Myr for different turbulence strengths
`alpha`, with the analytic steady state overlaid, to estimate alpha.

### Section 5: Grain size thresholds

```bash
cd Section_5
python section_5.py
```

Part 1 runs the solver at one radius for a few grain sizes (about 15 seconds each)
and checks the steady-state midplane dust-to-gas ratio against the analytic result.
Part 2 uses that analytic steady state to find the grain size needed for
`epsilon > 1` at 5, 10, 20 and 30 AU.

### Section 5.1: Resolution study

```bash
cd Section_5.1
python section_5_1.py
```

Shows how the number of grid points `N` affects the steady-state solution and
measures the convergence towards the analytic steady state.

### Verification

```bash
cd Verification
python verify_solver.py
```

Checks mass conservation, the exact advection-only midplane growth and the
analytic steady state. Prints PASS or FAIL for each check.

## Numerical scheme

Densities are stored at cell centres and fluxes on cell faces (finite volume).
The midplane is a face with zero flux, so the scheme conserves dust mass to
rounding error. Advection is first-order upwind; diffusion is explicit and
centred. The time step satisfies both the CFL and the diffusion limit.

## Interactive simulations

`Simulation/dust_settling_sim.html` (1D solver) and `Simulation/disc_3d_model.html`
(3D disc) open in a web browser.
