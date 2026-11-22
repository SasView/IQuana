"""
Example: from a 2D detector image to 1D curves (circular, sector, annular)

A 2D image is often most useful as a few 1D curves: the whole-detector
average, cuts along and across a field, and the intensity around a ring.
This script demonstrates how to:
1. Average a measured image over the whole detector and compare the result
   with the facility's own 1D reduction, on that file's Q points
2. Cut sectors along and across the field of a polarised (SANSPOL) image, and
   check the regions on the image with plot_cuts()
3. Measure the anisotropy around a ring with fit_sin2(), in three channels
4. Combine cuts of different files on one Q grid (bins=<a cut>)
5. Use a cut like any 1D dataset: a model fit, a joint fit of two sectors,
   and a Guinier analysis

Run from the repository root:  python examples/averaging_example.py
"""

import warnings

import numpy as np

from iquana import IQuana, MultiFitter, data_ops, examples, modelfree, set_verbosity

# The package picks ASCII spellings for these when the console cannot encode
# them, which keeps this script runnable on a legacy Windows terminal.
from iquana.console import DEGREE, INVERSE_ANGSTROM, OK, PHI


def banner(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


set_verbosity('quiet')

# ============================================================================
# Part 1: circular average, compared with the reduced 1D file
# ============================================================================
banner('Part 1: circular average of a SANS2D image')

sans2d = examples.load_fitter('sans2d_2d')  # beam stop and dI = 0 pixels masked
reduced = examples.load('canSAS_xml')  # the facility's 1D reduction of the same run

iq = sans2d.circular_average()  # default grid: bins as wide as the pixel pitch
print(f'{OK} Default grid: {iq.x.size} bins, {int(np.isfinite(iq.y).sum())} with data')

# bins=<a curve> lands on that curve's Q points, so the two can be divided.
on_file_grid = sans2d.circular_average(bins=reduced)
middle = (reduced.x > 0.03) & (reduced.x < 0.11) & np.isfinite(on_file_grid.y)
ratio = on_file_grid.y[middle] / reduced.y[middle]
print(
    f'Image / 1D file between Q = 0.03 and 0.11 {INVERSE_ANGSTROM}: '
    f'{ratio.min():.3f} to {ratio.max():.3f} (median {np.median(ratio):.3f})'
)
print('Below Q = 0.03 a bin of this image holds only a few pixels and scatters more.')

# ============================================================================
# Part 2: sectors along and across the field
# ============================================================================
banner('Part 2: sectors of a SANSPOL image (field horizontal)')

plus = examples.load_fitter('sanspol_plus')
along = plus.sector_average('horizontal')  # 0 deg +/- 10 deg, both sides of the beam
across = plus.sector_average('vertical')  # 90 deg
ring_plus = plus.annular_average(0.02, 0.04)

# The regions drawn on the image next to the curves: the quickest check that
# the angles mean what you think. Returned, not shown, in this script.
figure = plus.plot_cuts(along, across, ring_plus, show=False)
print(f'{OK} plot_cuts: {len(figure.data)} traces (image, outlines, curves)')

# All cuts of one image share one Q grid, so they subtract at once.
anisotropic = data_ops.subtract(across, along)
low_q = np.isfinite(anisotropic.y) & (anisotropic.x < 0.03)
print(
    f'across - along below Q = 0.03: mean {np.mean(anisotropic.y[low_q]):.3f} '
    '(the anisotropic part of this channel, not the magnetic scattering alone)'
)

# ============================================================================
# Part 3: the anisotropy around a ring
# ============================================================================
banner('Part 3: I(phi) in the ring 0.02 to 0.04 1/A, three channels')

unpol = examples.load_fitter('sanspol_unpol')
minus = examples.load_fitter('sanspol_minus')

fit_plus = ring_plus.fit_sin2()
fit_unpol = unpol.annular_average(0.02, 0.04).fit_sin2()
print(f'I+    : A = {fit_plus.a:.3f}, B = {fit_plus.b:.3f}, {PHI}0 = {fit_plus.phi0:.1f}{DEGREE}')
print(
    f'unpol : A = {fit_unpol.a:.3f}, B = {fit_unpol.b:.3f}, {PHI}0 = {fit_unpol.phi0:.1f}{DEGREE}'
)

# A free fit returns the direction of least intensity. In I- the intensity is
# lowest across the field, so the free fit points 90 deg away. Fix the
# direction found in I+ instead: B comes out negative.
ring_minus = minus.annular_average(0.02, 0.04)
free_minus = ring_minus.fit_sin2()
fixed_minus = ring_minus.fit_sin2(phi0=fit_plus.phi0)
print(
    f'I-    : free fit {PHI}0 = {free_minus.phi0:.1f}{DEGREE} (least intensity, across the field)'
)
print(f'I-    : {PHI}0 fixed to the I+ value: B = {fixed_minus.b:.3f} +/- {fixed_minus.b_err:.3f}')
print()
print(fit_plus.format_summary())

# ============================================================================
# Part 4: cuts of different files on one grid
# ============================================================================
banner('Part 4: combining the channels')

# Two files have their own pixel coordinates, so their default grids need not
# agree. bins=<a cut> puts a cut of the second file on the first one's grid.
magnetic = data_ops.subtract(
    unpol.sector_average('vertical', bins=across),
    unpol.sector_average('horizontal', bins=across),
)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # empty corner bins become NaN rows
    cross_term = data_ops.subtract(minus.sector_average('vertical', bins=across), across)
print(
    f'{OK} Magnetic term (unpolarised, across - along): {int(np.isfinite(magnetic.y).sum())} points'
)
print(f'{OK} Cross term (I- - I+, across the field): {int(np.isfinite(cross_term.y).sum())} points')

# ============================================================================
# Part 5: a cut is an ordinary 1D dataset
# ============================================================================
banner('Part 5: fitting the cuts')

one_d = IQuana()
one_d.set_data(along)
one_d.set_model('sphere')
one_d.set_param('radius', value=70, min=10, max=200, vary=True)
one_d.set_param('scale', value=0.002, min=1e-6, max=1, vary=True)
one_d.set_param('background', value=0.1, min=0, max=5, vary=True)
one_d.set_resolution('none')
result = one_d.fit()
print(f'Sphere fit along the field: radius = {result["parameters"]["radius"]["value"]:.1f} A')

# The two sectors see the same particles: share the size, keep the rest local.
joint = MultiFitter()
for name, cut in (('along', along), ('across', across)):
    joint.add(name, cut, model='sphere')
    joint[name].set_param('radius', value=70, min=10, max=200, vary=True)
    joint[name].set_param('scale', value=0.002, min=1e-6, max=1, vary=True)
    joint[name].set_param('background', value=0.1, min=0, max=5, vary=True)
    joint[name].set_resolution('none')
joint.share('radius')
joint_result = joint.fit(method='lm')
print(
    f'Joint fit of both sectors: shared radius = {joint_result.parameters["along.radius"].formatted} A'
)

# Model-free analyses take a cut as they take any 1D curve. A Guinier window
# needs q*Rg < 1.3, so it sits at the low-Q end of the unpolarised average.
guinier = modelfree.guinier(unpol.circular_average(), qmax=0.019)
print(
    f'Guinier on the unpolarised circular average: Rg = {guinier.rg:.1f} A ({guinier.n_points} points)'
)

print('\nDone.')
