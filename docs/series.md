# Series Fitting

A temperature scan, a kinetic run, a concentration or field series: many
datasets, one model, and the question is how the parameters change along the
series. `fit_series()` fits each dataset on its own with the setup of one
configured fitter and returns a table of the parameters against the series
variable.

```python
from iquana import IQuana

fitter = IQuana()
fitter.load_data('scan/T_020K.dat')
fitter.set_model('sphere')
fitter.set_param('radius',     value=45,   min=10,    max=100, vary=True)
fitter.set_param('scale',      value=0.02, min=0.001, max=0.1, vary=True)
fitter.set_param('background', value=0.01, min=0,     max=1,   vary=True)
fitter.fit()                                  # one good fit first (optional)

series = fitter.fit_series('scan/T_{temperature}K.dat')
```

```text
Series fit: 41 datasets · sphere · 3 free parameters · start=previous
    #0  T_020K.dat  temperature=20   χ²ᵣ 1.04   [1/41, ~40 s left]
   #16  T_052K.dat  temperature=52   χ²ᵣ 9.83   bound: radius at upper bound (100)
   #17  T_054K.dat  temperature=54   FAILED: No data points in Q range [0.03, 0.3]
✓ 41 datasets in 38 s: 38 ok, 2 flagged, 1 failed.   series.flagged() shows them.
```

```python
series                           # the table (in a notebook; print(series) elsewhere)
series.plot('radius')            # radius against temperature
series.flagged()                 # the rows that need a person
series.save_analysis('scan.json')
```

There is no second place to set a model and no new parameter syntax. The
configured fitter is the **template**: its model, composites, structure factor,
polydispersity, links, bounds, resolution, weighting and Q cut reach every
dataset, because each dataset is fitted by the ordinary `fit()` on a fresh copy
of it. A series of one file gives exactly the numbers of `fit()` on that file.
The template itself is not changed, so you can run `fit_series()` twice with
different settings and compare.

The Q cut is replayed as you asked for it: an end left open in `set_q_range()`
stays open, so each dataset uses its own end. A range the template got any
other way (limits carried by a dataset given to `set_data()`) is replayed as
its two numbers.

This is not `MultiFitter`. A series fits N independent problems; a
[simultaneous fit](multifit.md) fits one problem with parameters shared between
datasets.

## Which datasets

| `data` | Meaning |
|---|---|
| A file path | One dataset, one row. |
| A string with `*` or `{name}` | A pattern. Only these two are special: `?`, `[` and `]` are ordinary characters, `*` does not cross a directory, and `**` is the same as `*` (there is no recursive search). |
| A list | Paths, in-memory datasets (anything `set_data()` accepts, such as `data_ops` results), or a mixture, fitted in the order given. |

Each row has a **label**: a file's path relative to the common directory of the
files (just the file name when they share one directory), or an in-memory
dataset's title (`dataset_<i>` when it has none). Labels address rows, so they
must be unique.

## The series value

Each row has a value of the series variable. There are three ways to give it;
use one.

**In the file name.** A `{name}` placeholder reads a number from each file
name and names the variable:

```python
fitter.fit_series('scan/T_{temperature}K.dat')    # temperature = 20, 22, ...
fitter.fit_series('kinetics/run*_t{time}s.dat')   # a placeholder and a * together
```

The number may be signed or written with an exponent (`-3.5`, `1e-3`). A file
that matches the pattern but has no number there is an error that names it.

**As a list**, with a list of datasets:

```python
fitter.fit_series(files, values=[0.5, 1, 2, 5], variable='concentration')
```

A list of values is refused with a pattern: a pattern is sorted, so the
pairing of files and values would be in doubt.

**From file metadata.** A dotted path read from every dataset before the first
fit:

```python
fitter.fit_series('loki/run_*.nxs', values='sample.temperature')
```

| `values=` | Reads | Variable |
|---|---|---|
| `'sample.temperature'` | `data.sample.temperature` | `temperature` |
| `'sample.position.z'` | `data.sample.position.z` | `z` |
| `'meta_data.magnetic_field'` | `data.meta_data['magnetic_field']` | `magnetic_field` |

A file without a numeric value there is an error that lists the numeric
metadata the file does have, so the right path is easy to find:

```text
ValueError: run_0412.nxs has no value at 'sample.temperature' (it is None).
  Numeric metadata in this file: sample.thickness = 1, sample.transmission = 0.87
```

sasdata reads typed fields for temperature, thickness, transmission, position
and orientation. There is no typed field for magnetic field, pressure or time:
such values land in `meta_data` under a key that depends on the file layout.
Temperature is therefore the dependable case, and a field scan will usually
use the file-name placeholder. Check one of your own files with
`print(data.sample)` and `print(data.meta_data)`.

Units are not converted. When files record a unit for the value and the units
differ, the run is refused; when some files record no unit, the run says that
their consistency could not be checked. sasdata 0.12 does not pass on the units
of NXcanSAS (HDF5) files at all, so for those files the check never applies:
if a scan mixes units, convert the values yourself and pass `values=[...]`.

Every file is read for its value before the first fit, so with this form an
unreadable file stops the call (it could not be placed in the series). Remove
it, or pass the files as a list with `values=[...]`, where a broken file
becomes a failed row like any other.

**With no value**, the variable is `index`: 0, 1, 2, ...

**The variable's name** defaults to the placeholder name, the last segment of
the metadata path, `value` for a list, or `index`. `variable=` overrides it.
It names a column of the table, so it must be a Python identifier
(`temperature`, not `Temperature (K)`) and must not clash with a parameter
column such as `radius` or `radius_err`.

## Order

The order matters twice: `start='previous'` chains the fits in this order, and
the trend plot joins the points in it.

- A **list** is fitted in the order given.
- A **pattern** is sorted. `order='value'` (the default when there is a value)
  sorts by the series value, ties by file name. `order='name'` is natural
  file-name order (`run2` before `run10`), which is acquisition order when the
  run number comes first. Without a value, name order is the only order.

A hysteresis loop or a cool-then-heat scan needs name order. Sorted by value,
the two branches would interleave:

```python
fitter.fit_series('loop/run*_H{field}T.dat', order='name')
```

Repeated values are fine, and the trend plot draws the loop as a loop.

## Start values

| `start=` | Each fit starts from | Use when |
|---|---|---|
| `'previous'` (default) | The fitted values of the last usable row; the first row starts from the template. | Parameters drift along the series. |
| `'template'` | The template's values, every time. | The datasets are unrelated, or to cross-check a chained run. |

Only values are carried along. Bounds and vary flags always come from the
template.

A row is **usable as a start** when its fit completed, the engine did not
report non-convergence, and every varied value is finite and inside its
bounds. A failed row is not, and the next fit starts from the last usable row
instead. Each row records which row it started from.

A row resting *on* a bound is usable, but the parameters on the bound start
the next fit from the template's value, not from the bound. A start exactly on
a bound traps the default optimizer (bumps' amoeba never leaves it), so a scan
that crosses a bound and comes back would stay pinned there. A parameter that
really belongs on the bound, such as a background at zero, returns there in
one fit. The row says so in its warnings.

A chain can still be led astray by a fit that is wrong but looks successful.
The `jump` and `bound` flags are there to show that, and a second run with
`start='template'` is the check.

`method='differential_evolution'` builds its population from the bounds and
ignores the start values; the run says so once, and records no chain.

## While it runs

Single-fit reports are silenced. A line is printed for the first row (so you
see how long one fit takes), for every row that fails or is flagged during the
run, and at most every 10 s with progress and an estimate of the time left. The
row numbers are the table's. `set_verbosity('debug')` prints every row and
`set_verbosity('quiet')` none.

Warnings raised while a dataset is fitted are collected on its row instead of
being printed one by one; the summary counts them and `series.warnings()`
lists each text with the number of rows.

**Ctrl-C** stops the run and returns the rows finished so far, with a warning.
The fit in progress is discarded, also with methods that catch the interrupt
themselves. That needs the main thread: run from another thread, Ctrl-C is
not handled, and since Python's warning capture is process-wide, warnings that
other threads raise during the run can land in a row.

## The result

```text
Series fit · sphere · 41 datasets · variable: temperature · start=previous
38 ok, 2 flagged, 1 failed

 #  label       temperature  radius          scale                background         χ²ᵣ    flags
 0  T_020K.dat           20  45.12 ± 0.21    0.020100 ± 0.000300  0.0102 ± 0.0004    1.04
 1  T_022K.dat           22  45.30 ± 0.22    0.020000 ± 0.000300  0.0101 ± 0.0004    1.10
 ...
16  T_052K.dat           52  100 ± -         0.0031 ± 0.0009      0.0180 ± 0.0006    9.83   bound, jump
17  T_054K.dat           54  -               -                    -                  -      failed
```

The columns are the row number, the label, the series variable, every
parameter the template varies (active polydispersity widths included, link
followers not) with its uncertainty, reduced χ² and the flags. An uncertainty
that is not available shows as `-`. Fixed parameters are the same in every row
and stay in the template. Tables longer than 20 rows show the first and last
ten.

**Row numbers are identities.** `#` is the position in the original run and
never changes. Everything that takes a row takes its number or its label.

| Member | Gives |
|---|---|
| `len(series)`, `for row in series` | The rows. |
| `series[16]`, `series['T_052K.dat']` | One row, which describes itself when printed. |
| `series.variable`, `series.parameters` | The variable's name; the parameter columns. |
| `series.table()` | One flat dict per row: `row`, `label`, the variable, each parameter and `<name>_err`, `chisq`, `reduced_chisq`, `n_points`, `flags`, `edited`, `error`. |
| `series.to_dataframe()` | The same as a pandas DataFrame indexed by row (pandas is optional). |
| `series.flagged()` | The flagged rows (below). |
| `series.warnings()` | Each distinct warning text with its number of rows. |
| `series.plot(*parameters)` | Trend plot. |
| `series.plot_fits(rows=None)` | Data and fits for chosen rows. |
| `series.fitter(row)` | A normal `IQuana` for one row. |
| `series.update(row, fitter)` | Write a repaired fit back. |
| `series.save_table('scan.csv')` | The table as CSV, with each failure's reason. |
| `series.save_analysis('scan.json')` | Everything, reloadable. |
| `SeriesResult.load_analysis('scan.json')` | The reverse. |

The CSV method is `save_table()`, not `save_results()`: on a single fitter,
`save_results()` writes one fit's curve and residuals, which is a different
kind of file.

A row explains itself:

```text
>>> series[16]
Row 16 · T_052K.dat · temperature = 52
  started from: row 15 in the original run
  radius      100 ± -   radius at upper bound (100)
  scale       0.0031 ± 0.0009
  background  0.0180 ± 0.0006
  χ²ᵣ 9.83 · 212 points · flags: bound, jump
  warnings: (none)
  -> series.fitter(16) opens it as a normal IQuana
```

## Flags

A flag means "look at this row". It never removes a row or changes a number.

| Flag | When |
|---|---|
| `failed` | Loading, the Q cut or the fit raised. The row has no result; the message is kept. |
| `unconverged` | The engine reported that it did not converge. |
| `out_of_bounds` | A varied parameter ended outside its bounds (`method='leastsq'` ignores bounds). |
| `bound` | A varied parameter ended on a bound. |
| `jump` | A varied parameter moved much more than is usual in this series. |

`unconverged` can only come from the lmfit engine: bumps reports no verdict, so
"no flags" under bumps does not mean "all converged".

**The jump rule.** For each varied parameter, take the steps between adjacent
rows that both hold a converged fit (a failed or unconverged row breaks the
chain). A row is flagged when its step is more than 5 times the median step of
that parameter in the series *and* more than 3 times the combined uncertainty
of the two values (that second test is skipped where an uncertainty is not
available). Fewer than five steps flag nothing. The rule follows series order,
so it works on loops; it will flag a genuine phase transition, which is a
place to look anyway. It does not account for uneven spacing of the variable.
The thresholds are a first guess; tell us how they behave on your scans.

Reduced χ² is a column and a plot panel, not a flag: what counts as high
depends on the error bars of the instrument.

`series.flagged()` returns the flagged rows as a `SeriesResult` of their own,
with the reason in place of the parameter columns:

```text
>>> series.flagged()
Series fit · sphere · 3 of 41 rows flagged · variable: temperature · start=previous
0 ok, 2 flagged, 1 failed

 #  label       temperature  χ²ᵣ    why
16  T_052K.dat           52  9.83   radius at upper bound (100); radius jumped +54.9 (median step 0.2)
17  T_054K.dat           54  -      failed: No data points in Q range [0.03, 0.3]
31  T_082K.dat           82  1.21   radius jumped +12.1 (median step 0.2)
```

The row numbers are the original ones: `series.flagged()[16]` is row 16. It
prints, plots and exports like any series, but it is a read-only selection:
`update()` and `save_analysis()` belong to the full series.

## Plots

`series.plot()` draws every varied parameter, then reduced χ², as stacked
panels over the series variable; `series.plot('radius', 'scale')` chooses the
panels. Points are joined in series order and the line breaks at a failed or
unconverged row. Flagged rows are drawn as diamonds and give their reason on
hover; failed rows are marked × on the bottom axis.

`series.plot_fits()` overlays data and fit for up to nine rows spread over the
series. `rows=[3, 16, 31]` or `rows='flagged'` chooses others. A row that
cannot be drawn is named with its reason. If a row's saved result can no
longer be restored (its data file has changed), the model is drawn at the
stored values and labelled as a theory curve.

Both return plotly figures and follow the rules of every other IQuana plot.

## Drill down and repair

A flagged row needs a person. `series.fitter(row)` gives an ordinary `IQuana`
with the row's data, setup and result, so everything you know from single
fits applies:

```python
bad = series.fitter(16)
bad.plot_results()
bad.set_param('radius', value=60, max=200)
bad.fit(series.engine, series.method, **series.fit_options)   # the series' own fit
series.update(16, bad)
```

A plain `bad.fit()` uses the single-fit defaults. If those differ from what
the series used, `update()` accepts the row with a warning: its χ² and
uncertainties are not comparable with the other rows'.

| The row | `fitter(row)` gives |
|---|---|
| Fitted | The fitter as it was after the fit. |
| Loaded, but the Q cut or the fit failed | The dataset with the setup of the attempt, no result. |
| Its file never loaded | The template setup with the file loaded now: the load error again while the file is broken, the retry once it is fixed. |

`update()` checks three things before it replaces anything, and leaves the row
as it was if one fails:

1. the fitter holds a result that describes its current setup. Changing a
   bound and forgetting `fit()` gives *"Fit this dataset again before updating
   row 16."*;
2. its data is the row's data (1D, as for the run);
3. its model is the series' model. A different model belongs in a second
   series.

Values, bounds and vary flags may differ from the template. The table keeps
the template's columns. The row is marked `edited` (in the table and in the
saved file) and the flags of the whole series are recomputed. `edited` is a
marker, not a flag, so repairing the last problem empties `flagged()`.

`series.fitter(16).save_analysis(...)` or `.report(...)` hands one dataset to a
colleague as a self-contained file.

## Save and reload

```python
series.save_analysis('scan.json')
series = SeriesResult.load_analysis('scan.json')
```

Each row is stored as a complete single analysis, the same document
`IQuana.save_analysis()` writes, so every guarantee of that file holds per row:
the result is checked against its setup, a changed data file is reported, and
custom models need `allow_custom_models=True`. Data files are referenced, not
copied, with a path relative to the series file as well as the absolute one,
so a project moved together with its data still finds it. In-memory datasets
are embedded (roughly 20 to 50 kB per 200-point row, so a thousand of them make
a file of tens of megabytes).

Loading reads the JSON only: the table, the flags and the trend plot need no
data files. A row's data is found and checked when you open it with `fitter()`
or draw it with `plot_fits()`. An interrupted run saves and reloads as such
("18 of 41 processed, interrupted"); there is no resume, but the remaining
files can be fitted with a second call.

`IQuana.load_analysis()` on a series file, or `SeriesResult.load_analysis()` on
a single analysis, says which call to use.

## Limits

- **1D data only** in this version. 2D series, parallel execution and Bayesian
  sampling per dataset are planned.
- **One model per series.** Different models, Q cuts or bounds for some
  datasets go through the drill-down, or into a second series.
- **No trend fitting.** `table()` and `to_dataframe()` give the numbers; fit
  the radius against temperature with numpy or scipy.
- **Decimal points written as letters** (`T_20p5K`) are not read by the
  placeholder; pass `values=` for such files.

A runnable walkthrough is in `examples/series_fitting_example.py` and
`notebooks/series_fitting.ipynb`.
