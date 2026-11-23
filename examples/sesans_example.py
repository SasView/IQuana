"""
Example: Fitting SESANS and Reduced SEMSANS Data

This script demonstrates how to:
1. Load a .ses file and fit it with an ordinary I(Q) model
2. Restrict the fit to a range of spin-echo lengths
3. Fit time-of-flight data with a structure factor
4. Build a dataset from arrays (reduced SEMSANS data) with data_ops.sesans()
5. See the sensitivity warning for structure smaller than the first step
6. Save the analysis, reopen it, and export the fit

The data are the two SESANS examples that ship with sasdata: 2 um polystyrene
spheres measured at Delft (one wavelength) and PMMA hard spheres measured at
ISIS (time of flight). See docs/sesans.md for the background.
"""

import os
import shutil
import warnings

import numpy as np
import sasdata

from iquana import IQuana, data_ops

SES_DIR = os.path.join(os.path.dirname(sasdata.__file__), 'example_data', 'sesans_data')
DELFT = os.path.join(SES_DIR, 'sphere2micron.ses')
ISIS = os.path.join(SES_DIR, 'sphere_isis.ses')

WORKDIR = os.path.abspath('sesans_example_output')
if os.path.isdir(WORKDIR):
    shutil.rmtree(WORKDIR)
os.makedirs(WORKDIR)


def sphere_setup(fitter, radius):
    """A dilute sphere with unit contrast: radius and scale (phi * drho^2) free."""
    fitter.set_model('sphere')
    fitter.set_param('radius', value=radius, min=radius / 5, max=radius * 5, vary=True)
    fitter.set_param('scale', value=0.01, min=1e-4, max=1, vary=True)
    fitter.set_param('sld', value=1.0, vary=False)
    fitter.set_param('sld_solvent', value=0.0, vary=False)


# ============================================================================
# Part 1: A .ses file, fitted like any other dataset
# ============================================================================

print('=' * 80)
print('Part 1: 2 um polystyrene spheres (Delft)')
print('=' * 80)

fitter = IQuana()
fitter.load_data(DELFT)
print(f'SESANS data: {fitter.is_sesans}')
sphere_setup(fitter, radius=9000)
result = fitter.fit()
radius = result['parameters']['radius']
print(f'Radius: {radius["value"]:.0f} +- {radius["stderr"]:.0f} A (nominal 10 000 A)')

# Linear axes: the depolarisation is negative.
fitter.plot_results()

# ============================================================================
# Part 2: The fit range is in spin-echo length
# ============================================================================

print('\n' + '=' * 80)
print('Part 2: Fit only 5000 to 30000 A')
print('=' * 80)

fitter.set_q_range(5000, 30000)
ranged = fitter.fit()
print(f'Points fitted: {ranged["n_points"]} of {len(fitter.data.x)}')
fitter.reset_q_range()
fitter.fit()  # back to the whole range, so the saved result in Part 6 matches it

# ============================================================================
# Part 3: Time of flight, and a structure factor
# ============================================================================

print('\n' + '=' * 80)
print('Part 3: PMMA hard spheres (ISIS, time of flight)')
print('=' * 80)

tof = IQuana()
tof.load_data(ISIS)
sphere_setup(tof, radius=1000)
tof.set_structure_factor('hardsphere')
tof.set_param('scale', value=0.3, min=0.01, max=0.7, vary=True)
tof_result = tof.fit()
print(
    f'Radius: {tof_result["parameters"]["radius"]["value"]:.0f} A, '
    f'reduced chi-squared {tof_result["reduced_chisq"]:.2f}'
)

# ============================================================================
# Part 4: Arrays, such as reduced SEMSANS data
# ============================================================================

print('\n' + '=' * 80)
print('Part 4: A dataset from arrays with data_ops.sesans()')
print('=' * 80)

# Reduced SEMSANS data is a table of spin-echo length, visibility ratio Vs/V0
# and its error, usually read with np.loadtxt. Here the ratio is made from the
# Delft depolarisation, P/P0 = exp(t * lambda^2 * depolarisation).
loaded = data_ops.load(DELFT)
thickness, lam = 0.2, 2.11
ratio = np.exp(thickness * lam**2 * loaded.y)
ratio_error = ratio * thickness * lam**2 * loaded.dy

semsans = data_ops.sesans(
    loaded.x,  # spin-echo length, A
    polarisation=ratio,  # P/P0, or Vs/V0 for SEMSANS
    error=ratio_error,
    wavelength=lam,  # A: one number, or one per point
    thickness=thickness,  # cm
    acceptance=0.0168,  # rad: see docs/sesans.md for SEMSANS
)
arrays = IQuana()
arrays.set_data(semsans)
sphere_setup(arrays, radius=9000)
array_result = arrays.fit()
print(f'Radius from the arrays: {array_result["parameters"]["radius"]["value"]:.0f} A')

# ============================================================================
# Part 5: Structure smaller than the first step
# ============================================================================

print('\n' + '=' * 80)
print('Part 5: The sensitivity warning')
print('=' * 80)

# Spheres of 1000 A on the Delft grid, whose first step is 1172 A: the transform
# cannot resolve them, and the fit says so while still returning its result.
# The data are the model itself, with 5 % error bars.
model = IQuana()
model.set_data(semsans)
sphere_setup(model, radius=1000)
model.set_param('scale', value=0.13)
curve = model.calculate()
small_data = data_ops.sesans(
    loaded.x,
    depolarisation=curve,
    error=np.full_like(curve, 0.05 * np.abs(curve).max()),
    wavelength=lam,
    acceptance=0.0168,
)
small = IQuana()
small.set_data(small_data)
sphere_setup(small, radius=1200)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    small.fit()
for warning in caught:
    if 'integration limit' in str(warning.message):
        print(f'Warning: {warning.message}')

# ============================================================================
# Part 6: Save, reopen, export
# ============================================================================

print('\n' + '=' * 80)
print('Part 6: Save and reopen')
print('=' * 80)

analysis = os.path.join(WORKDIR, 'polystyrene.json')
fitter.save_analysis(analysis)
reopened = IQuana.load_analysis(analysis)
print(f'Reopened with its result: {reopened.fit_result is not None}')

table = os.path.join(WORKDIR, 'polystyrene.csv')
fitter.save_results(table)
with open(table, encoding='utf-8') as handle:
    header = next(line for line in handle if not line.startswith('#'))
print(f'CSV columns: {header.strip()}')

shutil.rmtree(WORKDIR)
