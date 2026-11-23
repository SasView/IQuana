# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 0.7

### Added

- **SESANS and reduced SEMSANS data.** A `.ses` file loads and fits with the
  usual workflow (`load_data`, `set_model`, `fit`, `plot_results`): sasmodels
  turns any I(Q) model into ln(P/P₀)/(tλ²) through its Hankel transform. See
  `docs/sesans.md`.
  - `data_ops.sesans(spin_echo_length, polarisation=... | depolarisation=...,
    error=, wavelength=, acceptance=, thickness=)` builds a dataset from
    arrays, such as reduced SEMSANS visibility ratios.
  - `IQuana.is_sesans` and `SeriesResult.is_sesans` say which kind of data is
    held.
  - `set_q_range()` takes spin-echo lengths for SESANS data, and the range,
    masks and NaN points are honoured by every engine, `fit_bayesian()`, the
    plots and the export. sasmodels ignores them for SESANS data; IQuana
    selects the points itself while the transform keeps the whole grid, so a
    kept point's curve does not depend on the range.
  - A fit warns when the curve depends on the transform's integration limit
    (structure much smaller than the first spin-echo step), and still returns
    its result.
  - Linear axes, SESANS labels, a "Transform" line instead of the resolution
    mode, and SESANS column names in `save_results()`
    (`spin_echo_length,depolarisation,error,fit,residual`).
  - `fit_series()` fits SESANS series; a series holds one kind of data.
  - `notebooks/sesans_fitting.ipynb` and `examples/sesans_example.py`.

- **2D to 1D: circular, sector and annular averages** (roadmap item 3.3):
  - `IQuana.circular_average()`, `sector_average(phi, half_width=10,
    mirror=True)` and `annular_average(qmin, qmax, bins=36)`, using the
    fitter's pixel selection, Q range and resolution setting; the same three
    as functions in `data_ops`, with `mask=` and `mask_convention=`, for a bare
    dataset.
  - The I(Q) cuts are fit-ready `Data1D` objects: they go into `set_data()`,
    `MultiFitter.add()`, `fit_series()`, `modelfree`, P(r) and
    `data_ops.subtract()` as they are, and carry the image's metadata and a
    provenance record.
  - One Q grid per image, from the local pixel pitch by default (merged
    multi-distance images get one width per pitch), so every cut of one image
    has the same Q values. `bins=` takes a number (with `log=True`), edges, an
    earlier cut (to put a second file on the first one's grid) or any `Data1D`
    (to land on the points of a reduced 1D file). Empty bins, and bins a mask
    or a Q limit cuts through, are NaN and masked.
  - dQ from the file's radial width or the fitter's pinhole setting, plus the
    spread of |Q| inside each bin.
  - `AnnularAverage` (I(φ)) with `fit_sin2()`, which returns an
    `AnisotropyFit`: A, B and φ₀ of I = A + B·sin²(φ − φ₀), with errors, a
    summary, a `FitReport` and a plot. Both exported from the package root.
  - `plot_cuts()` (fitter method and `data_ops` function): the regions
    outlined on the image next to the curves.
  - Named warnings: `uncertainties_from_scatter`, `zero_uncertainty_pixels`,
    `one_sided`, `sides_differ`, `partial_ring`, `uneven_ring`,
    `weak_anisotropy`, and `shared_pixels` from arithmetic on cuts of one
    image that share pixels.
  - `notebooks/averaging_2d.ipynb` and `examples/averaging_example.py`; a
    "From 2D to 1D" section, with "Differences from SasView", in the user
    guide.

### Changed

- `plot_model()`, `plot_results()`, `compare()` and
  `plot_posterior_predictive()` take `log_scale=None` by default, meaning log
  axes for I(Q) data (as before) and linear for SESANS data. An explicit
  `log_scale=True` on SESANS data is refused.
- A SESANS analysis is saved as schema version 4 (`data.measurement`, and the
  wavelengths and acceptance of an in-memory dataset). I(Q) analyses stay at
  version 3, so 0.6 keeps reading them. A SESANS analysis saved by an earlier
  version loads without its fit result, which ignored the fit range.
- SESANS data refuses what does not apply to it: a resolution mode other than
  `'data'`, a weighting other than `'dI'` (on the setters and on loading), a
  free `background`, a custom `q=` grid, `MultiFitter` and `data_ops`
  arithmetic.
- `data_ops` arithmetic refuses a first operand without dQ when the second has
  it, with a message, instead of failing inside sasdata with a `TypeError`.
- The 1D data summary says "151 (104 with data)" for a curve with NaN rows.
- The "1D only" messages of `MultiFitter`, `fit_series()`, `data_ops`,
  `modelfree` and P(r) point at the averages.

## [0.6.0] - 2026-11-20

### Added

- **Model-free first-look analysis**, `iquana.modelfree` (roadmap item 3.2):
  - `guinier()`: Rg and I(0), or the cross-section of a rod (`kind='rod'`) or
    the thickness of a sheet (`kind='sheet'`). Without `qmax` the window grows
    until q·R reaches `qr_limit` (1.3 for the standard law). Warns when the
    data do not reach the Guinier plateau, and when ln I bends down
    (repulsion) or up (elongation, polydispersity, aggregation) beyond what a
    compact particle shows.
  - `power_law()`: I − B = A·q⁻ⁿ over a window, with a fixed background (log
    fit, with a `low_signal` warning near the background) or a fitted one.
  - `porod()`: the Porod constant and the flat background in one fit, with a
    free-exponent check, and `specific_surface(delta_sld)`.
  - `kratky()`: q²·I, or the dimensionless form with the Guinier-law peak
    marked.
  - `invariant()`: Q* with Guinier, Porod or power-law extrapolations; the
    volume fraction (both Babinet roots) from `delta_sld`; the specific
    surface by the Porod route and, with a known `volume_fraction`, by the
    scale-free route. The uncertainty is propagated once from the intensities
    through every fit, so shared points and a fitted background count once.
  - Every result has `format_summary()`, `to_dict()`, `save_csv()`, `plot()`
    and named warnings; the three fits also have `report()`, a `FitReport`
    like a model fit's. Chained calls must use one background and one
    dataset, and refuse otherwise. Slit-smeared, 2D and SESANS data are
    refused.
  - `notebooks/first_look.ipynb` and `examples/modelfree_example.py`.
- The result types and `ModelFreeError` are exported from the package root.
- `FitReport.notes`: free-text notes, rendered in every format and included
  in `to_dict()`. Reports without notes are unchanged.
- **2D data.** `load_data()` and `set_data()` accept 2D detector images.
  `plot_model()` and `plot_results()` draw data, model and residual
  heatmaps; `fit()`, `fit_bayesian()`, `calculate()`, `save_results()`,
  `save_analysis()` and `report()` work on them. `set_q_range()` selects an
  annulus in |Q|. The fitter's `is_2d` property says which kind is loaded.
- Pixel masks for 2D data: `mask_ring()`, `mask_box()`, `mask_invalid()`,
  `set_mask()`, `reset_mask()` and `get_mask()`. The selection is fitter
  state, never written to the dataset's own mask, and is saved with the
  analysis. The pixel at |Q| = 0 is always excluded.
- `set_data(data, mask_convention=...)` for 2D objects whose mask uses the
  other convention.
- 2D resolution: `'data'` uses the radial and azimuthal widths, `'pinhole'`
  sets both to `dq_over_q·|Q|`; `set_resolution(..., accuracy=...)` sets the
  2D oversampling.
- **Magnetism.** `enable_magnetism(beam=..., efficiency_in=..., efficiency_out=...)`,
  `supports_magnetism()`, `get_magnetic_parameters()` and
  `is_magnetism_enabled()` expose sasmodels' magnetic parameters for 2D
  data, with beam presets for unpolarised, SANSPOL and polarisation-analysis
  experiments. A fit refuses a polarisation channel whose magnetisation can
  reach zero (sasmodels switches kernels there) and reports magnetic
  parameters the model does not respond to.
- `examples.simulate_2d()`, and the 2D examples `p123_2d`, `sans2d_2d`,
  `sanspol_plus`, `sanspol_minus` and `sanspol_unpol`.
- A fit warns when orientation parameters vary on 1D data, where they have no
  effect.
- `notebooks/magnetic_2d_fitting.ipynb` and `examples/magnetic_2d_example.py`.
- **Generic Scattering Calculator.** `iquana.GenericScatteringCalculator` and
  the `iquana.structure` module expose the
  [SincCity](https://easyscience.github.io/SincCity/) package: I(q) of atoms
  (PDB, GRO), pixels (SLD) or mesh cells (VTK) by the Debye formula, a 3D view
  of the points, contrast series, and a settings record that replays a
  calculation.
  - `IQuana.set_data()`, `MultiFitter.add()` and
    `IQuana.load_analysis(data=...)` accept a `ScatteringProfile` directly.
  - The fit guard names the weighting and resolution to set for a computed
    curve.
  - `set_verbosity()` covers SincCity's messages, and anything that captures
    the `iquana` logger receives them.
  - Walkthroughs: `examples/structure_calculator_example.py` and
    `notebooks/structure_calculator.ipynb`; example structures in
    `examples/data/`.
- `save_analysis()` writes the arrays of an in-memory 1D dataset (a
  `data_ops` result, a simulated or computed curve) into the analysis file, so
  `load_analysis(path)` reopens it without `data=`. A template
  (`include_result=False`) embeds no arrays and still needs `data=`, as does
  an in-memory 2D dataset. The `schema` block also records the SincCity
  version.

### Changed

- P(r) inversion and the model-free analyses share one data preparation,
  `iquana.data.prepare`. P(r) behaves as before; its "points dropped"
  warning now names the P(r) inversion. `InsufficientDataError` is now
  defined there (`__module__` is `iquana.data.prepare`); every import path
  still works.
- The analysis file schema is version 3. It adds `data.dimension`, a `mask`
  section, the arrays of an in-memory 1D dataset (`data.arrays`) and, only
  when used, `configuration.magnetism` and `resolution.accuracy`. Versions 1
  and 2 load as before, with their results; an IQuana 0.5 refuses a version 3
  file with its "newer schema" message.
- `examples.simulate()` rejects magnetic parameters, which it used to accept
  and silently ignore.
- `MultiFitter` refuses 2D data with a clear message before adding it.
- The minimum versions are now `sasmodels>=1.0.12` and `sasdata>=0.11`, the
  oldest the tests run against.
- `sinccity` is a required dependency.

### Fixed

- `set_data()` no longer modifies the object it is given. It used to write
  `qmin`, `qmax` and `mask` onto it, and `set_q_range()` then changed that
  same object, so fitters given one dataset shared a Q range: a second fitter
  took the first one's restricted range as its full range, and narrowing
  either changed both. Each fitter now keeps a shallow copy that shares the
  arrays but has its own limits and mask. Code that read `qmin`, `qmax` or
  `mask` back from the object after `set_data()` should read `fitter.data`
  instead.

## [0.5.0] - 2026-10-28

This release fixes three silent fitting errors. Fits that use a structure
factor, the LMFit engine, or polydispersity can give different results from
0.4. Check the **Changed** and **Fixed** sections before you upgrade.

### Added

- Fits check every varying parameter before starting. A fit is refused, with
  the parameter named, if the parameter starts outside its bounds or if its
  bounds meet (`min == max`) or cross (`min > max`). The check runs for both
  engines and for `fit_bayesian()`.
  ([#1](https://github.com/SasView/IQuana/pull/1))
- `fit(engine='lmfit', method='leastsq')` warns when a fitted value ends up
  outside its bounds. ([#1](https://github.com/SasView/IQuana/pull/1))
- `set_pd_param()` warns when it is asked for a width or `vary=True` after
  polydispersity was switched off with `enable_polydispersity(False)`.
  ([#16](https://github.com/SasView/IQuana/pull/16))
- An integration test suite runs every `examples/*_example.py` script end to
  end. Run it with `pixi run test-integration` or
  `pytest tests/integration -m integration`. A GitHub workflow runs it on
  Linux, Windows and macOS for release pull requests, every Sunday, and on
  demand. ([#14](https://github.com/SasView/IQuana/pull/14))
- `simulated_sans_data.csv` and `simulated_sans_data_with_resolution.csv` in
  `tests/data`, used by the notebooks and examples.

### Changed

- **The LMFit engine's default method is now `'least_squares'`**, which keeps
  parameters inside their bounds. It was `'leastsq'`, which ignores bounds.
  Pass `method='leastsq'` to get the old behaviour.
  ([#1](https://github.com/SasView/IQuana/pull/1))
- **`set_pd_param()` switches polydispersity on** when it is given a width
  above zero or `vary=True`. A separate `enable_polydispersity(True)` call is
  only needed to switch it back on after switching it off.
  ([#16](https://github.com/SasView/IQuana/pull/16))
- Under the `'dI'` weighting mode, a missing or empty `dI` column is treated
  as all zeros, the same way sasdata represents a file without `dI`.
  ([#12](https://github.com/SasView/IQuana/pull/12))
- Analysis files saved by 0.4 that use a structure factor load with the
  `radius_effective_mode` kernel switch corrected (see **Fixed**) and with a
  warning. Their saved fit result is not restored, because the old kernel
  computed it; refit after loading.
  ([#13](https://github.com/SasView/IQuana/pull/13))
- The notebooks and the test helpers no longer add the repository to
  `sys.path`. Install the package (`pip install -e .` or `pixi install`)
  before running them. ([#2](https://github.com/SasView/IQuana/pull/2))
- The API reference for `ParameterManager` and the polydispersity constants
  (`PD_DEFAULTS`, `PD_DISTRIBUTION_TYPES`) is generated from the source code,
  so it lists the current method names.
  ([#3](https://github.com/SasView/IQuana/pull/3))

### Removed

- `run_tests.py`. Run the tests with `pytest` or `pixi run test`.
  ([#2](https://github.com/SasView/IQuana/pull/2))

### Fixed

- **A structure factor ignored the `radius_effective` parameter.** sasmodels'
  product kernel has its own `radius_effective_mode` switch, and its default
  computes the effective radius from the form factor. IQuana now sets the
  switch so `radius_effective` is used, both when it is free
  (`'unconstrained'`) and when it is linked to `radius` (`'link_radius'`).
  Results differ most for `'unconstrained'` and for non-spherical form
  factors. ([#13](https://github.com/SasView/IQuana/pull/13),
  [#9](https://github.com/SasView/IQuana/issues/9))
- **`set_pd_param()` settings were silently left out of the fit** until
  `enable_polydispersity()` was called.
  ([#16](https://github.com/SasView/IQuana/pull/16),
  [#9](https://github.com/SasView/IQuana/issues/9))
- The LMFit engine could return values outside the bounds set with
  `set_param()`, such as a negative radius.
  ([#1](https://github.com/SasView/IQuana/pull/1))
- The LMFit engine failed with a `TypeError` from sasmodels on data that has
  no `dI` column (`dy=None`). It now fits such data with unit weights and a
  warning, as it already did for zero uncertainties.
  ([#12](https://github.com/SasView/IQuana/pull/12))
- Removing a structure factor did not restore whether polydispersity had been
  switched on or off by the user, which could produce a misleading
  "switched off" warning.

## [0.4.0] - 2026-10-23

- First release as IQuana, carried over from SANS-fitter 0.4.

[0.5.0]: https://github.com/SasView/IQuana/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/SasView/IQuana/releases/tag/v0.4.0
