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

Produces Figure 2 of the report (this takes about a minute). Set `N = 400` and
`t_hat_final = 50500000.0` to get Figure 1.

### Section 4.1: Estimating alpha

```bash
cd Section_4.1
python section_4_1.py
```

Compares dust profiles for different turbulence strengths `alpha` to estimate alpha.

### Section 5: Grain size thresholds

```bash
cd Section_5
python section_5.py
```

Runs a simulation for one radius and one grain size. Running every radius and grain
size at once would take too long, so to scan more grain sizes at a given radius,
uncomment entries in `a_list`. The final plot shows the threshold grain size needed
for a dust-to-gas ratio above 1 at different radii.

### Section 5.1: Resolution study

```bash
cd Section_5.1
python section_5_1.py
```

Shows how the number of spatial grid points `N` affects the solution.
