"""
Example: model-free first look with iquana.modelfree

This script demonstrates how to:
1. Simulate a polydisperse sphere dataset on absolute scale, with known truth
2. Fit Porod first, so every later step subtracts the same background
3. Fit a Guinier law with an automatic Q window, and read its checks
4. Draw the dimensionless Kratky plot
5. Compute the invariant, the volume fraction and the specific surface
6. Fit a power law over a chosen window
7. Use the results as starting values for a model fit
"""

import math

from iquana import IQuana, examples, modelfree

# ============================================================================
# Part 1: simulate polydisperse spheres (R = 50 Å, 20 % width, φ = 1 %)
# ============================================================================
# With measured data you would start at data_ops.load() or fitter.data instead.
radius, width, phi, delta_sld = 50.0, 0.2, 0.01, 3.0
data = examples.simulate(
    'sphere',
    radius=radius,
    radius_pd=width,
    scale=phi,
    sld=4.0,
    sld_solvent=1.0,
    background=0.001,
    qmin=0.002,
    qmax=0.6,
    npoints=300,
    noise=0.01,
    seed=1,
)

# ============================================================================
# Part 2: Porod at high Q gives the Porod constant and the flat background
# ============================================================================
p = modelfree.porod(data, qmin=0.15)
print(p.format_summary())
print(f'Simulated background: {data.truth["background"]}')

# ============================================================================
# Part 3: Guinier at low Q, with the Porod background subtracted
# ============================================================================
g = modelfree.guinier(data, background=p.background)
print('\n' + g.format_summary())
print(g.report())  # the same kind of report as a model fit
g.plot(data)

# ============================================================================
# Part 4: dimensionless Kratky plot
# ============================================================================
k = modelfree.kratky(data, guinier=g)  # takes the Guinier result's background
print('\n' + k.format_summary())
k.plot()

# ============================================================================
# Part 5: invariant, volume fraction and specific surface
# ============================================================================
# delta_sld is in 1e-6 1/Å², like every SLD in IQuana. Passing the known
# volume fraction as well gives the scale-free specific surface and a check
# of the absolute scale.
inv = modelfree.invariant(data, low_q=g, high_q=p, delta_sld=delta_sld, volume_fraction=phi)
print('\n' + inv.format_summary())
print(f'Expected Q* = {2 * math.pi**2 * delta_sld**2 * phi * 1e-4:.4g}')
inv.plot()
inv.save_csv('invariant.csv')

# ============================================================================
# Part 6: a power law over a chosen window
# ============================================================================
pl = modelfree.power_law(data, qmin=0.15, qmax=0.6, fit_background=True)
print('\n' + pl.format_summary())

# ============================================================================
# Part 7: starting values for a model fit
# ============================================================================
fitter = IQuana()
fitter.set_data(data)
fitter.set_model('sphere')
fitter.set_param('radius', value=g.sphere_radius, min=1, max=500)
fitter.set_param('background', value=p.background, min=0, max=1)
# Rg of a polydisperse sample is intensity-weighted (dominated by the large
# spheres), so the Guinier radius is a starting value above the number mean.
print(f'\nStarting radius from Guinier: {g.sphere_radius:.1f} Å (number mean {radius} Å)')

print('\n✓ Example completed successfully!')
