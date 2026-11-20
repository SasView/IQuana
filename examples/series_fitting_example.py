"""
Example: Fitting a Temperature Scan (Series Fitting)

This script demonstrates how to:
1. Set up and fit one dataset, then apply the same setup to a whole scan
   with fit_series(), reading the temperature from the file names
2. Read the result: the table, the flagged rows, the trend plot
3. Cross-check a chained run (start='previous') against independent fits
4. Open a flagged row as a normal fitter, repair it, and write it back
5. Save the series, reopen it, and export the table as CSV
6. Fit a list of in-memory datasets with values given explicitly

The data are simulated with a known answer: spheres whose radius grows slowly
with temperature and jumps at a transition near 50 K. One file is broken, and
one frame has a radius beyond the bounds the template allows.
"""

import os
import shutil

import numpy as np

from iquana import IQuana, SeriesResult, data_ops, examples

WORKDIR = os.path.abspath('series_fitting_example_output')
if os.path.isdir(WORKDIR):
    shutil.rmtree(WORKDIR)
SCAN = os.path.join(WORKDIR, 'scan')
os.makedirs(SCAN)

TEMPERATURES = list(range(20, 62, 2))


def true_radius(temperature):
    """Slow growth, then a 10 Å jump at the transition, then slow growth again."""
    radius = 40.0 + 0.2 * (temperature - 20)
    return radius + 10.0 if temperature >= 50 else radius


# ============================================================================
# Part 1: The scan
# ============================================================================

print('=' * 80)
print('Part 1: A simulated temperature scan')
print('=' * 80)

for i, temperature in enumerate(TEMPERATURES):
    radius = 95.0 if temperature == 40 else true_radius(temperature)  # one odd frame
    data = examples.simulate(
        'sphere', radius=radius, scale=0.01, background=0.001, noise=0.02, seed=i, npoints=80
    )
    path = os.path.join(SCAN, f'T_{temperature:03d}K.dat')
    np.savetxt(path, np.column_stack([data.x, data.y, data.dy]))

# The detector software crashed while writing one file.
with open(os.path.join(SCAN, 'T_036K.dat'), 'w') as handle:
    handle.write('# truncated\n')

print(f'Wrote {len(TEMPERATURES)} files to {SCAN}')

# ============================================================================
# Part 2: One good fit, then the whole scan
# ============================================================================

print()
print('=' * 80)
print('Part 2: Fit one dataset, then the scan')
print('=' * 80)

fitter = IQuana()
fitter.load_data(os.path.join(SCAN, 'T_020K.dat'))
fitter.set_model('sphere')
fitter.set_param('radius', value=45, min=10, max=80, vary=True)
fitter.set_param('scale', value=0.02, min=1e-4, max=1, vary=True)
fitter.set_param('background', value=0.01, min=0, max=1, vary=True)
fitter.set_resolution('none')  # the simulated files carry no dQ column
fitter.set_q_range(qmin=0.01)  # trim the lowest points; the upper end stays open
fitter.fit()

# The {temperature} placeholder finds the files, reads each temperature and
# sorts by it. Every fit starts from the previous good one (start='previous').
series = fitter.fit_series(os.path.join(SCAN, 'T_{temperature}K.dat'))

# ============================================================================
# Part 3: Reading the result
# ============================================================================

print()
print('=' * 80)
print('Part 3: The table, the flags, the trend')
print('=' * 80)

print(series)
print()
print(series.flagged())
print()

# One row describes itself: where its fit started, its values, its flags.
print(series['T_040K.dat'])

# Every varied parameter and reduced chi-squared against temperature. The jump
# at 50 K and the odd frame stand out; the broken file is marked on the axis.
series.plot()
series.plot('radius')
series.plot_fits(rows='flagged')

# ============================================================================
# Part 4: Cross-check the chain
# ============================================================================

print()
print('=' * 80)
print("Part 4: Is the chain trustworthy? Compare with start='template'")
print('=' * 80)

# start='previous' carries each fit's values to the next one. A fit that is
# wrong but looks fine could lead the rest astray; independent fits from the
# template's values are the check.
independent = fitter.fit_series(os.path.join(SCAN, 'T_{temperature}K.dat'), start='template')
chained = {row['label']: row['radius'] for row in series.table()}
disagree = [
    row['label']
    for row in independent.table()
    if row['radius'] is not None and abs(row['radius'] - chained[row['label']]) > 0.5
]
print('Chained and independent radii differ by more than 0.5 Å for:', disagree or 'no row')

# ============================================================================
# Part 5: Repair a flagged row
# ============================================================================

print()
print('=' * 80)
print('Part 5: Open a flagged row, repair it, write it back')
print('=' * 80)

odd = series['T_040K.dat']
print(f'Row {odd.index}: reduced chi-squared {odd.reduced_chisq:.0f}, flags: {odd.why()}')
print(f'Its radius, {odd.estimate("radius")[0]:.1f} Å, is pressed against max=80.')

repair = series.fitter(odd.index)  # a normal IQuana: data, setup and result
repair.plot_results()
repair.set_param('radius', max=150)  # the bound was too tight for this frame
repair.fit()
series.update(odd.index, repair)

# The repaired fit is good, and the frame is still flagged 'jump': its radius
# really is far from its neighbours'. Flags point at rows to look at; they do
# not say a fit is wrong.
print(series[odd.index])

# The broken file cannot be repaired here. Once a good copy exists,
# series.fitter(row) loads it again and the same repair path applies.
broken = series['T_036K.dat']
print(f'\nRow {broken.index} failed: {broken.error}')

# ============================================================================
# Part 6: Save, reopen, export
# ============================================================================

print()
print('=' * 80)
print('Part 6: Save the series, reopen it, export the table')
print('=' * 80)

series_file = os.path.join(WORKDIR, 'scan.json')
series.save_analysis(series_file)
reopened = SeriesResult.load_analysis(series_file)
print(f'Reopened: {len(reopened)} rows, {len(reopened.flagged())} of them flagged')

# Any row opens as an ordinary fitter, also after a reload.
row_fitter = reopened.fitter(5)
print(f'Row 5 radius: {row_fitter.params["radius"]["value"]:.2f} Å')

reopened.save_table(os.path.join(WORKDIR, 'scan.csv'))

# Fit the trend yourself: the table gives the numbers. Leave out the flagged
# rows, and stay below the transition.
table = [row for row in reopened.table() if not row['flags'] and row['temperature'] < 50]
x = np.array([row['temperature'] for row in table])
y = np.array([row['radius'] for row in table])
slope = np.polyfit(x, y, 1)[0]
print(f'Radius growth below the transition: {slope:.3f} Å/K (simulated: 0.200 Å/K)')

# ============================================================================
# Part 7: In-memory datasets with explicit values
# ============================================================================

print()
print('=' * 80)
print('Part 7: A concentration series from in-memory datasets')
print('=' * 80)

# Background-subtracted curves from data_ops are fitted as they are; the
# concentrations are given as a list, paired with the datasets in order.
solvent = examples.simulate('sphere', scale=0.0, background=0.002, noise=0.0, npoints=80)
solvent.dy = np.full_like(solvent.y, 1e-4)
concentrations = [0.5, 1.0, 2.0, 5.0]
subtracted = []
for i, concentration in enumerate(concentrations):
    sample = examples.simulate(
        'sphere',
        radius=42.0,
        scale=0.004 * concentration,
        background=0.002,
        seed=10 + i,
        npoints=80,
    )
    curve = data_ops.subtract(sample, solvent)
    curve.title = f'c = {concentration} mg/ml'
    subtracted.append(curve)

dilution = IQuana()
dilution.set_model('sphere')
dilution.set_param('radius', value=45, min=10, max=80, vary=True)
dilution.set_param('scale', value=0.01, min=1e-5, max=1, vary=True)
dilution.set_param('background', value=0.0, vary=False)
dilution.set_resolution('none')
by_concentration = dilution.fit_series(subtracted, values=concentrations, variable='concentration')
print(by_concentration)
by_concentration.plot('scale')

print(f'\nOutput written to {WORKDIR}')
