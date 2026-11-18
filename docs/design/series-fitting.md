# Design: batch (series) fitting

*Status: proposal, revision 2, 2026-11-18, against IQuana 0.6.0. Target release: 0.8.*
*Implements item 3.1 of the roadmap ("Batch (series) fitting").*
*Revision 2 takes in three design reviews (CC, GPT, DS). Appendix B lists what
was adopted and what was not.*

## 1. Summary

Add one method, `IQuana.fit_series()`, and one result class, `SeriesResult`.

The user sets up and fits one curve exactly as today. The configured fitter is
then the template for the whole series:

```python
fitter = IQuana()
fitter.load_data('scan/T_020K.dat')
fitter.set_model('sphere')
fitter.set_param('radius',     value=45,   min=10,    max=100, vary=True)
fitter.set_param('scale',      value=0.02, min=0.001, max=0.1, vary=True)
fitter.set_param('background', value=0.01,                     vary=True)
fitter.fit()                                  # one good fit first (optional)

series = fitter.fit_series('scan/T_{temperature}K.dat')
```

```text
Series fit: 41 datasets · sphere · 3 free parameters · start=previous
   #0  T_020K.dat  temperature=20   χ²ᵣ 1.04
  #16  T_052K.dat  temperature=52   χ²ᵣ 9.83   bound: radius (max)
  #17  T_054K.dat  temperature=54   FAILED: No data points in Q range [0.03, 0.3]
✓ 41 datasets in 38 s: 38 ok, 2 flagged, 1 failed.   series.flagged() shows them.
```

```python
series.plot('radius')            # radius against temperature
series.flagged()                 # the rows that need a person
series.save_analysis('scan.json')
```

`T_020K.dat` matches the pattern, so it is fitted again as row 0. That is
intended: it gets the same engine, options and place in the chain as every
other row.

There is no new fitter class, no second place to set a model, and no new
parameter syntax. Everything the template knows (model, composites, structure
factor, polydispersity, links, bounds, resolution, weighting, Q cut) applies
to every dataset because the loop calls the public `fit()` once per dataset.

**Decisions taken with the project owner** (2026-11-18)

| Question | Decision |
|---|---|
| Entry point | `fitter.fit_series()` on a configured `IQuana`, not a `BatchFitter` class. |
| Series value | One of: a `{name}` placeholder in the file pattern, a list of numbers, or a metadata path such as `'sample.temperature'`. Otherwise the row index. |
| Order of a pattern | `order='value'` (default when a value exists) or `order='name'` (natural file-name order). |
| Default start | `start='previous'`. `start='template'` is the alternative. |
| Progress output | First row, every flagged or failed row, and a progress line with an ETA at most every 10 s. |
| Units of the variable | None in phase 1. `variable` is a plain name. |
| Bulk refit of flagged rows | After phase 1. Phase 1 stores what it needs (`start_row`). |
| Phase 1 scope | 1D data, serial, optimiser fits, drill-down and repair of single rows. |
| Phase 2 scope | 2D datasets, parallel execution, Bayesian sampling per dataset (section 9). |

## 2. Goals and non-goals

**Goals (phase 1)**

1. Fit N 1D datasets independently with the setup of one configured fitter.
2. Attach a series variable (temperature, time, concentration, field) to each
   dataset with as little typing as possible.
3. Return a table of parameters, uncertainties and χ² against that variable,
   and plot any parameter against it.
4. Flag datasets where the fit failed, did not converge, left its bounds,
   ended on a bound, or jumped away from its neighbours.
5. Save and reload the whole series as JSON; export the table as CSV.
6. Let the user open, repair and write back a single row.

**Non-goals (phase 1)**

- Shared or constrained parameters across datasets. That is `MultiFitter`.
- Different models, Q cuts or bounds per dataset. The drill-down (3.9) is the
  escape hatch for the few datasets that need special treatment.
- Fitting a trend (radius against temperature) to a function. `table()` and
  `to_dataframe()` give the numbers; numpy and scipy do the rest.
- A progress bar widget, or any new dependency.
- 2D data, parallel execution and Bayesian sampling per dataset. These are
  phase 2 and are designed in section 9 so that phase 1 does not block them.

## 3. User interface

### 3.1 Signature

```python
def fit_series(
    self,
    data,                  # pattern, path, or list of paths / datasets
    *,
    values=None,           # list of numbers, or a metadata path such as 'sample.temperature'
    variable=None,         # name of the series variable; overrides the derived one
    order=None,            # pattern only: 'value' or 'name'
    start='previous',      # or 'template'
    engine='bumps',
    method=None,
    **kwargs,              # passed to fit(), as today
) -> SeriesResult: ...
```

Everything after `data` is keyword-only. `engine`, `method` and `**kwargs`
mean exactly what they mean in `fit()`.

### 3.2 Which datasets

| `data` | Meaning |
|---|---|
| An existing file path | One dataset. `fit_series('scan/T_020K.dat')` works and gives one row. |
| A string with `*` or `{name}` | A pattern. Only these two are special; `?`, `[abc]` and `**` are literal characters. |
| A list | Paths, in-memory datasets (anything `set_data()` accepts), or a mixture, in the order given. |

**Labels.** A file's label is its path relative to the common directory of the
files (the file name, in the usual one-directory case). An in-memory dataset's
label is its title, or `dataset_<i>` when it has none. Labels must be unique;
a duplicate is an error before any fit.

### 3.3 The series value

There are three ways to give the value, and exactly one may be used. Two at
once is an error.

**In the file name.** A single `{name}` placeholder finds the files, extracts
the number and names the variable:

```python
fitter.fit_series('scan/T_{temperature}K.dat')
# T_020K.dat, T_022K.dat, ...  ->  temperature = 20, 22, ...
```

The placeholder matches a plain number (`20`, `-3.5`, `1e-3`). A file that
matches the pattern but has no number there is an error naming the file. More
than one placeholder is an error. A pattern may also contain `*`.

**As a list.** Only with a list of datasets, so the pairing is never in doubt
(a list of values with a pattern is an error, because the pattern is sorted):

```python
fitter.fit_series(files, values=[0.5, 1, 2, 5], variable='concentration')
```

**From file metadata.** A string is a dotted path read from each loaded
dataset. A segment is an attribute, or a key when the object is a dict:

```python
fitter.fit_series('loki/run_*.nxs', values='sample.temperature')
```

| `values=` | Reads | Default `variable` |
|---|---|---|
| `'sample.temperature'` | `data.sample.temperature` | `temperature` |
| `'sample.position.z'` | `data.sample.position.z` | `z` |
| `'meta_data.magnetic_field'` | `data.meta_data['magnetic_field']` | `magnetic_field` |

All files are read for their value before the first fit. A file whose value is
missing, `None` or not finite is an error that names the file and lists the
numeric metadata that is there:

```text
ValueError: run_0412.nxs has no value at 'sample.temperature' (it is None).
  Numeric metadata in this file: sample.thickness = 1.0, sample.transmission = 0.87,
  meta_data['magnetic_field'] = 0.25
```

What sasdata provides today: typed fields for temperature, thickness,
transmission, position and orientation. There is no typed field for magnetic
field, pressure or time; such values land in `meta_data` under a key that
depends on the file layout. So the dependable case is temperature, and a field
scan will in practice use the file-name placeholder. The docs will say this,
and the facility examples stay provisional until checked on one real LOKI and
one real SKADI file.

Units are not handled. One reviewer found that the unit string does not
survive the reader in sasdata 0.12.2 (`temperature_unit` came back `None` for
files written in K and in °C). So: when units are present and differ between
files, that is an error; when they are absent, the run logs once that
consistency could not be checked.

The file is read twice with this form (once for the value, once by
`load_data()`), which keeps provenance handling unchanged. For 1D files that
is milliseconds each.

**With no value** the variable is `index`: 0, 1, 2 ...

**The variable name.** Each form gives a default: the placeholder name, the
last segment of a metadata path, `'value'` for a plain list, `'index'`
otherwise. `variable=` overrides it. The name is the key in `table()`, the
DataFrame column and the CSV header, so it must be a valid Python identifier
and must not collide with any column of the finished table, including the
generated ones (`radius_err`). It is checked before any fit.

### 3.4 Order

Order matters because `start='previous'` follows it and the trend plot joins
points in it.

- A **list** is fitted in the order given. Passing `order=` with a list is an
  error.
- A **pattern** is sorted. `order='value'` sorts by the series value; ties are
  broken by natural file name, then full path. `order='name'` is natural
  file-name order (`run2` before `run10`), which is acquisition order when the
  run number comes first.
- The default is `'value'` when the pattern has a value (placeholder or
  metadata) and `'name'` when it has none. `order='value'` without a value is
  an error.

`order='name'` is what a hysteresis loop or a cool-then-heat scan needs:

```python
fitter.fit_series('loop/run*_H{field}T.dat', order='name')
```

Repeated values are legal. Sorting such a scan by value would interleave the
two branches.

### 3.5 Start values

| `start=` | Each fit starts from | Use when |
|---|---|---|
| `'previous'` (default) | The fitted values of the last usable row. The first starts from the template. | Parameters drift along the series. |
| `'template'` | The template's values, every time. | Datasets are unrelated, or to cross-check a chained run. |

Only values are carried over. Bounds and vary flags always come from the
template.

A row is **usable as a start** when its fit completed, the engine did not
report non-convergence, and every varied value is finite and inside its
bounds. Otherwise the next fit starts from the last usable row (or the
template).

- A row resting **on** a bound is usable. A background that really goes to
  zero should chain. This is a decision, not an oversight.
- A row **outside** its bounds is not usable. `engine='lmfit',
  method='leastsq'` ignores bounds and can end there; chaining from it would
  make every later fit fail the start check.

Each row records `start_row`: the row whose values seeded it in the original
run, or `None` for the template.

**Methods that ignore the start.** scipy's `differential_evolution` builds
its population from the bounds and is not given the start values. With
`start='previous'` the run says so once:

```text
Note: method='differential_evolution' starts from the bounds, not from the
previous row. start='previous' has no effect; the rows are independent.
```

A chain can still be led astray by a fit that is wrong but looks successful.
The `jump` and `bound` flags exist to show that, and re-running with
`start='template'` is the check.

### 3.6 What the user sees while it runs

Per-fit reports are silenced. A row is printed when:

1. it is the first row (the run has started; this is how long one fit takes);
2. it is flagged at fit time or failed;
3. at least 10 s have passed since the last printed line. That line carries
   `[done/total, ~time left]`.

Row numbers in the log are the table's row numbers, starting at 0.

```text
Series fit: 1000 datasets · sphere · 3 free parameters · start=previous
    #0  frame_0000.dat  index=0     χ²ᵣ 1.03                [1/1000, ~9 min left]
  #118  frame_0118.dat  index=118   χ²ᵣ 1.07                [119/1000, ~7 min left]
  #163  frame_0163.dat  index=163   χ²ᵣ 4.20   bound: scale (min)
  #201  frame_0201.dat  index=201   FAILED: could not read file
✓ 1000 datasets in 8 min 12 s: 981 ok, 17 flagged, 2 failed.   series.flagged() shows them.
  warnings: 1 distinct (1000 rows).   series.warnings() lists them.
```

`set_verbosity('debug')` prints every row; `'quiet'` prints none. The 10 s
interval is a module constant.

**Warnings** raised while a row is loaded and fitted are collected on the row
and not printed one by one. The bound and non-convergence warnings are not
stored again; they are already flags. The summary counts distinct warning
texts, and `series.warnings()` lists each text with the number of rows.

**Ctrl-C** stops the run and returns the rows finished so far, with a warning
giving the count. The row in progress is discarded. See 6.5.

### 3.7 The result: `SeriesResult`

In a notebook `series` renders as a table; `print(series)` gives the same as
text. More than 20 rows are shown as the first and last ten, with a line
saying how to get all of them.

```text
Series fit · sphere · 41 datasets · variable: temperature · start=previous
38 ok, 2 flagged, 1 failed

  #  label        temperature   radius          scale              background        χ²ᵣ    flags
  0  T_020K.dat            20   45.12 ± 0.21    0.0201 ± 0.0003    0.0102 ± 0.0004   1.04
  1  T_022K.dat            22   45.30 ± 0.22    0.0200 ± 0.0003    0.0101 ± 0.0004   1.10
 ...
 16  T_052K.dat            52   100.0 ± —       0.0031 ± 0.0009    0.0180 ± 0.0006   9.83   bound, jump
 17  T_054K.dat            54   —               —                  —                 —      failed
 ...
```

**Columns** are the row number, the label, the series variable, every
parameter the **template** varies with its uncertainty, reduced χ², and the
flags. "Varied" is read from the result as "not fixed": it includes active
polydispersity widths and excludes linked followers. An uncertainty of `None`
or zero on a varied parameter means "not available" and is shown as `—`.
Fixed parameters are the same in every row and stay in the template.

**Row numbers are identities.** `#` is the position in the original run and
never changes. Every accessor that takes a row takes its number or its label.

| Member | Returns |
|---|---|
| `len(series)`, iteration | The rows. |
| `series[16]`, `series['T_052K.dat']` | One `SeriesRow`, by row number or label. |
| `series.variable` | Name of the series variable. |
| `series.table()` | `list[dict]`, one flat dict per row: `row`, `label`, the variable, `radius`, `radius_err`, ..., `chisq`, `reduced_chisq`, `n_points`, `flags`, `edited`, `error`. |
| `series.to_dataframe()` | The same as a pandas DataFrame. pandas is imported here only. |
| `series.flagged()` | The rows carrying a flag (3.7.1). |
| `series.warnings()` | Distinct warning texts with row counts. |
| `series.plot(*parameters)` | Trend plot (3.8.1). |
| `series.plot_fits(rows=None)` | Data and fit curves (3.8.2). |
| `series.fitter(row)` | A normal `IQuana` for one row (3.9). |
| `series.update(row, fitter)` | Replace a row with a repaired fit (3.9). |
| `series.save_table('scan.csv')` | The table as CSV, including the failure reason. |
| `series.save_analysis('scan.json')` | Everything, reloadable (section 5). |
| `SeriesResult.load_analysis('scan.json')` | The reverse. |

The CSV method is `save_table`, not `save_results`: `IQuana.save_results()`
writes one fit's curve and residuals, which is a different file.

#### 3.7.1 `flagged()`

`flagged()` returns a `SeriesResult` holding only the flagged rows, so
printing, plotting, `to_dataframe()` and `save_table()` work on it unchanged.
Its text form replaces the parameter columns with the reason:

```text
>>> series.flagged()
Series fit · sphere · 3 of 41 rows flagged · variable: temperature

   #  label        temperature   radius         χ²ᵣ    why
  16  T_052K.dat            52   100.0 ± —      9.83   radius at upper bound (100); radius jumped +54.9 (median step 0.2)
  17  T_054K.dat            54   —              —      failed: No data points in Q range [0.03, 0.3]
  31  T_082K.dat            82   61.30 ± 0.40   1.21   radius jumped +12.1 (median step 0.2)
```

Row numbers are the original ones: `series.flagged()[16]` is row 16 and
`series.flagged()[0]` is a `KeyError`.

A filtered result is a **read-only selection**. It keeps the flags the full
series computed. `update()` and `save_analysis()` on it raise and say to call
them on the full series. That keeps one owner for the rows and for the
neighbour-based `jump` flag.

#### 3.7.2 A row explains itself

```text
>>> series[16]
Row 16 · T_052K.dat · temperature = 52
  started from: row 15 in the original run (start='previous')
  radius       100.0 ± —        at upper bound (100)
  scale        0.0031 ± 0.0009
  background   0.0180 ± 0.0006
  χ²ᵣ 9.83 · 212 points · flags: bound, jump
  warnings: (none)
  → series.fitter(16) opens it as a normal IQuana
```

### 3.8 Plots

#### 3.8.1 Trend plot

`series.plot()` draws every varied parameter, and reduced χ² at the bottom, as
stacked panels over a shared x axis. `series.plot('radius', 'scale')` selects
panels.

```text
 radius (Å)    |            o
           60  |         o     ◇        o = fitted value with error bar
           50  | o  o  o                ◇ = flagged row (hover shows why)
               +---------------------
 χ²ᵣ           |            ◇
               | o  o  o  o      o  o
               +---------×-----------
                 20  30  40  50  60   temperature     × = failed (hover: reason)
```

- Points are joined in series order, so a loop is drawn as a loop.
- The line is broken at a failed or unconverged row.
- Failed rows are marked with `×` on the x axis of the bottom panel.
- An unavailable uncertainty draws no error bar.

#### 3.8.2 Curves and fits

`series.plot_fits()` overlays data and fit for up to nine rows spread evenly
over the rows that have a fit. `rows=[3, 16, 31]` or `rows='flagged'` chooses
others. It reuses `DatasetPanel` and `plot_multi(overlay=True)`; the legend
entry carries label and value (`T_052K.dat · 52`).

A row that cannot be drawn is named with its reason, also when none can be. If
a row's saved result cannot be restored (the data file changed), the model is
evaluated at the stored parameters and the curve is labelled as a theory
curve, not a fit.

### 3.9 Drill down and repair

```python
series.flagged()                 # row 16: bound, jump
bad = series.fitter(16)          # a normal IQuana with data, setup and result
bad.plot_results()
bad.set_param('radius', value=60, max=200)
bad.fit()
series.update(16, bad)           # write the repaired fit back
```

**What `fitter(row)` returns**

| The row | You get |
|---|---|
| Fitted | The fitter as it was after the fit. |
| Loaded, then the Q cut or the fit failed | The fitter with that dataset and the setup of the attempt, no result. If the cut selected no points the dataset is at its full range; the error names the rejected cut. |
| The file never loaded | The template setup with the file loaded now. It raises the load error again if the file is still broken. This is also the retry path once the file is fixed. |

**What `update(row, fitter)` checks, before replacing anything**

1. The fitter holds a fit result that describes its current setup. Changing a
   bound and forgetting `fit()` gives: *"Fit this dataset again before
   updating row 16."* (This is the `result_omitted` check `save_analysis()`
   already makes.)
2. The data is the row's data: the array fingerprint equals the one the row
   was fitted on (`array_fingerprint_now`). For a row that never loaded, the
   file path and dataset selector must match instead.
3. The model is the same: expression, component names and structure factor.
   A different model is refused, with a pointer to a second series.

Values, bounds and vary flags may change freely. The table keeps the
template's columns: a parameter that the repaired row fixes shows its value
with no `±`, and a parameter that only the repaired row varies is not a column
(it is visible through `fitter(16)`).

After `update()` the row is marked **edited**, and flags are recomputed for
the whole series. `edited` is a marker, not a flag: it is shown, and it is
stored, but `flagged()` does not count it, so repairing the last problem
empties `flagged()`.

An update replaces one row. Rows that were chained from it keep their numbers;
`start_row` therefore describes the original run.

`series.fitter(i).save_analysis(...)` or `.report(...)` is the hand-over path
for one dataset.

## 4. Flags

A flag means "look at this row". It never removes a row or changes a number.
Flags are not stored; they are recomputed from the rows on load and after
every `update()`.

| Flag | Condition | Source |
|---|---|---|
| `failed` | Loading, the Q cut or the fit raised. No result; the message is kept. | The loop. |
| `unconverged` | The engine reported `converged is False`. | `FitResultContract.converged`. |
| `out_of_bounds` | A varied parameter ended outside its bounds. | `ParameterManager.start_values_outside_bounds()`. |
| `bound` | A varied parameter ended on a bound. | `FitResultContract.on_bounds`. |
| `jump` | A varied parameter moved much more than is usual in this series. | Below. |

`unconverged` can only appear with the lmfit engine: bumps reports no verdict.
The user docs will say that "no flags" under bumps does not mean "all
converged".

**The jump rule.** For each varied parameter, take the absolute steps between
**adjacent** rows that both have a converged fit. A step across a failed or
unconverged row is not taken. Row *i* is flagged when its step from row *i−1*
is both

- more than 5 times the median step of that parameter in the series, and
- more than 3 times the combined uncertainty of the two values (skipped when
  an uncertainty is not available).

It needs at least five steps; below that, nothing is flagged. The two
thresholds are module constants. They are a first guess and have not been
tried on real scans.

The rule works on series order, so it is correct for loops. It will flag a
real phase transition, which is intended. It does not account for uneven
spacing of the series variable.

Reduced χ² is a column and a plot panel, not a flag (see Appendix B).

## 5. Persistence

Each row is stored **as** a single analysis.

```json
{
  "schema": {"format": "iquana-series", "version": 1, "written_by": "0.8.0"},
  "variable": "temperature",
  "start": "previous",
  "engine": "bumps",
  "method": "amoeba",
  "fit_options": {"steps": 2000},
  "planned": 41,
  "processed": 41,
  "interrupted": false,
  "template": { "...": "single-analysis document, data = null, no result" },
  "rows": [
    {
      "label": "T_020K.dat",
      "value": 20.0,
      "source": "T_020K.dat",
      "start_row": null,
      "edited": false,
      "error": null,
      "warnings": [],
      "analysis": { "...": "single-analysis document, or null if the file never loaded" }
    }
  ]
}
```

- **Only what cannot be derived is stored.** Parameters, χ², point counts and
  all flags are rebuilt from `analysis.result` on load. A file therefore
  cannot hold flags that contradict its numbers, and the jump constants can
  change without migrating files.
- **No new result schema.** `rows[i].analysis` is written by
  `persistence.build_analysis_dict()` and read by the existing reader. Number
  encoding, the check that a result still describes its setup, the warning
  for a changed data file and the custom-model opt-in all hold per row.
- **The table needs no data files.** Loading a series reads the JSON only. A
  data file is resolved and verified when a row is opened with `fitter()` or
  drawn with `plot_fits()`. If it has changed, the row's numbers stay
  available and the message says why the curve could not be restored.
- **`source`** is the path that was attempted (`null` for an in-memory
  dataset). It exists so that a row whose file never loaded can be retried.
- **`template`** has its data section removed. A template that had a file
  loaded would otherwise name one dataset as "the" data of the series.
- **`fit_options`** holds the `**kwargs` given to `fit_series()` when they are
  JSON-able. If they are not, the run warns that the file will not record
  them.
- **`planned`, `processed`, `interrupted`** let a reopened partial run say
  "18 of 41 processed, interrupted". There is no resume.
- **Relative paths.** A row built during the run has no relative path, because
  the save location is not known yet. `save_analysis()` adds it. To do so it
  finds each file the way the reader does (relative to where the series was
  loaded from, then the absolute path) and writes the path relative to the new
  location. So a series that was moved with its data and saved again still
  points at the data.
- **Envelope checks are shared.** The format, schema-version and structure
  checks of the single-file reader are applied to the series file and to every
  row document. A row written by a newer IQuana is not accepted silently.
- **Wrong loader.** `IQuana.load_analysis('scan.json')` raises
  `AnalysisFileError`: *"This is a series file. Use
  SeriesResult.load_analysis()."*
- **Size.** A file-backed row is a few kilobytes. An in-memory row embeds its
  arrays, roughly 20 to 50 kB for 200 points, so a thousand in-memory frames
  give tens of megabytes. The docs will say so.

`SeriesResult.load_analysis()` takes `allow_custom_models`, as
`IQuana.load_analysis()` does, and applies it whenever a row is opened.

**Out-of-bounds values must reload.** Today a saved analysis whose parameter
value lies outside its bounds cannot be loaded at all
(`parameters._validated_param` raises). After a `leastsq` fit that is exactly
what gets saved. In a series, one such row would make the file unusable. The
reader will accept such a value **with a warning**. It then reproduces the
state the live fitter was in, and `fit()` still refuses to start from it with
its existing message. This also fixes the same failure for single analyses.
The reported optimum is never clamped.

## 6. How it works

### 6.1 The loop

```python
source = template
for number, (label, value, item) in enumerate(items):
    if stop_requested:
        break
    worker = source._copy_setup()          # setup and current values; no data, no result
    row = _fit_one(worker, number, label, value, item, q_request, engine, method, kwargs)
    rows.append(row)
    if start == 'previous' and row.usable_as_start:
        source = worker                    # carries the fitted values forward
```

A fresh worker per row means nothing from row *k−1* (data, provenance, result)
can leak into row *k*. `_copy_setup()` shares the compiled kernel, so it is
cheap. `fit()` never changes bounds or vary flags, so a worker copied from the
previous worker still has the template's.

### 6.2 One row

`_fit_one` runs inside `warnings.catch_warnings(record=True)`, from loading to
building the row, and inside one `try`:

1. `worker.load_data(path)` or `worker.set_data(obj)`.
2. Apply the template's Q cut (6.3). If it leaves no points, `set_q_range()`
   raises and **the row ends here** as `failed`. It must not fall through to a
   fit over the full range, which would look like a success.
3. `worker.fit(engine, method, **kwargs)`.
4. Build the row from the worker: the analysis document, `start_row`, the
   collected warnings.

Any `Exception` makes a `failed` row. Its `source` comes from `item`, never
from the worker. If step 1 succeeded the row still gets an analysis document
(without a result); if not, `analysis` is `None`.

Because step 3 is the public `fit()`, a series of one file gives the same
numbers as a single fit of it.

### 6.3 The Q cut is replayed as the user asked for it

The Q range lives on the dataset and is reset by `load_data()`, so the loop
must re-apply it. What it re-applies has to be the **request**, not the
resolved range. Today `set_q_range(qmin=0.03)` on data ending at 0.20 is
recorded as `{qmin: 0.03, qmax: 0.20000000000000004}`. Replaying that on a
frame reaching 0.40 would silently drop its upper half.

- `IQuana` keeps `_q_range_request`: the `(qmin, qmax)` given to the last
  successful `set_q_range()`, with `None` for an end that was not given. It is
  cleared by `reset_q_range()` and when data is replaced.
- The loop calls `worker.set_q_range(*request)` on each dataset, so an open
  end resolves against that dataset. A typed end holds on every dataset.
- `persistence._fit_range_section()` writes the request for new files. The
  schema already allows `null` for an end, and the reader already passes it to
  `set_q_range()`, so there is no schema change. Older files keep the two
  numbers they recorded.

That last point also fixes `load_analysis(path, data=other)` for single
analyses, which has the same fault.

### 6.4 Validation before the first fit

A mistake in the call should cost no fits. The data-independent checks of
`fit()` move into `IQuana._check_fit_setup(engine)`, called by both `fit()`
and `fit_series()`, so the two cannot drift apart:

| Check | Where |
|---|---|
| Engine name; scipy available for lmfit | `_check_fit_setup` |
| Composite model with a non-bumps engine | `_check_fit_setup` |
| **At least one free parameter** (new; counts active PD widths) | `_check_fit_setup` |
| Start values inside bounds | `_check_fit_setup` |
| Scale degeneracy (warning) | `_check_fit_setup` |
| Uncertainties; model against data | per row, in `fit()` |

`_check_fit_setup` does not require data: the template may have none. The
free-parameter check is new for `fit()` too. Without it an all-fixed model
fails inside bumps with `max() iterable argument is empty`.

`fit_series()` adds its own: a model is set; at least one dataset; unique
labels; one value source; `values` length; the variable name; `order` and
`start` values; a 2D template (phase 1). A 2D **file** in a pattern is found
when it is loaded and becomes a failed row, so files are not read twice just
to check their dimension.

There is no "stop after N identical failures" rule. Failures that are the same
for every row are caught by the checks above; the rest belong to their rows.

### 6.5 Interrupt

The loop must stop on Ctrl-C even when the engine swallows it. Two reviewers
report that bumps' `de` method and DREAM catch `KeyboardInterrupt` and return
normally; bumps was not installed in the environment this revision was written
in, so this is recorded as reported and is covered by a test with a real
interrupt.

For the duration of the run, when called from the main thread, the loop
installs a SIGINT handler that sets `stop_requested` and then calls the
previous handler. The loop checks the flag after each fit: the row in progress
is dropped (an optimisation cut short is not a result), no further row starts,
and the partial result is returned with a warning. The previous handler and
the log level are restored on every exit path.

### 6.6 The template is not modified

`fit_series()` works on copies. Afterwards the template has the same data,
parameters and result as before. It needs a model; it does not need data or a
fit of its own.

### 6.7 Data classes

```python
@dataclass
class SeriesRow:
    index: int                     # row number in the original run; never changes
    label: str
    value: float
    source: str | None             # path attempted; None for an in-memory dataset
    analysis: dict | None          # None when the dataset never loaded
    error: str | None
    warnings: tuple[str, ...]
    start_row: int | None
    edited: bool = False
```

`parameters`, `chisq`, `reduced_chisq`, `n_points` and `flags` are derived
from `analysis` and cached. `SeriesResult` holds the rows and the top-level
fields of section 5. Neither holds a fitter, a kernel or a dataset.

## 7. Changes to the code

**New**

| File | Contents |
|---|---|
| `src/iquana/series.py` | `fit_series()`, item resolver, `_fit_one()`, flags, `SeriesRow`, `SeriesResult` with rendering, CSV, save and load. One module to start; split off the result classes if it grows unwieldy, following `multifit.py` / `multi_results.py`. |
| `src/iquana/series_plotting.py` | Trend plot; `plot_fits` as a call into `multi_plotting`. |
| `tests/test_series.py` | Section 8. |
| `docs/series.md`, `examples/series_fitting_example.py`, `notebooks/series_fitting.ipynb` | One end-to-end story: fit a scan, inspect a failure, repair a row, reopen the saved series. |

**Changed**

| File | Change |
|---|---|
| `fitter.py` | `fit_series()`: thin method delegating to `series.fit_series`. |
| `fitter.py` | `_copy_setup()`: a new `IQuana` sharing the kernel, with copies of the parameter manager, resolution and weighting, and no data. `MultiFitter.add_fitter()` is **not** changed to use it: that method also copies data, keeps provenance and enforces its weighting rule. |
| `fitter.py` | `_check_fit_setup(engine)` split out of `fit()`, with the new free-parameter check (6.4). |
| `fitter.py` | `_q_range_request`, set by `set_q_range()`, cleared by `reset_q_range()` and on new data (6.3). |
| `fitter.py` | Build the post-fit report only when it will be logged: `if logger.isEnabledFor(logging.INFO)`. Today the report is built and rendered on every fit even when silenced, which a series multiplies by N. |
| `persistence.py` | `fitter_from_document(document, base_path, data=None, ...)` split out of `read_analysis()`. |
| `persistence.py` | `_fit_range_section()` writes the Q request (6.3). |
| `persistence.py` | `read_document()` names the right loader for a series file; envelope checks reusable for nested documents (section 5). |
| `modeling/parameters.py` | `_validated_param()` accepts a value outside its bounds with a warning (section 5). |
| `console.py` | A context manager that raises the log level for a block and restores it. |
| `__init__.py` | Export `SeriesResult`. |
| `docs/multifit.md`, `README.md`, `CHANGELOG.md`, `mkdocs.yml` | Replace "batch fitting is not supported"; add the page. |

Four of these change single-fit behaviour and get their own commits and
changelog lines: the free-parameter check, the Q request in saved files, the
relaxed reader, and the lazy report.

**Dependencies:** none added.

## 8. Tests

On simulated data from `examples.simulate()` with a known parameter ramp.

*The fit*

1. Truth recovery, for both start modes and both engines.
2. A series of one file equals `IQuana.fit()` on that file (deterministic
   method).
3. The template is identical before and after.
4. Q cut, resolution, weighting, polydispersity, a composite model and a
   structure factor each reach every row.
5. All-fixed template raises before any fit, from `fit()` and `fit_series()`;
   a template whose only free parameter is a PD width is accepted.
6. `differential_evolution` with `start='previous'` emits its note once.
7. The report is not built under `set_verbosity('quiet')` (count calls).

*Items, values, order*

8. A single path gives one row. Placeholder extraction; natural sort; list
   order kept; `?` and `[` are literal.
9. Errors before any fit: no match; two placeholders; non-numeric match;
   wrong `values` length; `values` list with a pattern; two value sources;
   duplicate labels; a variable name that is not an identifier or collides
   with a column, including `radius_err`.
10. `order='name'` with repeated values keeps file order; `order=` with a
    list raises; `order='value'` without a value raises.
11. Metadata values on simulated NXcanSAS files: read and sorted by; a missing
    value raises and lists what is there; differing units raise when present.

*Q cut*

12. A one-sided cut keeps the full other end on a wider frame; a two-sided cut
    holds on a wider frame; a rejected cut leaves the previous range; the
    request survives save and load; an old file's two numbers are honoured.
13. A frame with no points in the cut is `failed` with `n_points` `None`, not
    fitted over its full range.

*Failures and the chain*

14. An unreadable file and an out-of-range frame mid-series: those rows fail,
    the others are fitted, and the next row starts from the last usable one.
15. A `leastsq` row outside its bounds: flagged `out_of_bounds`, not used as a
    start, and the series saves and reloads with every row openable.
16. A missing file: `fitter(i)` raises the load error; after the file is
    restored `fitter(i)` loads, `fit()` and `update()` succeed.
17. Flags: a bound hit and a discontinuity are flagged at the right rows; a
    smooth ramp has none; fewer than five steps never flag `jump`; no step is
    taken across a failed row.
18. Interrupt: a mocked `KeyboardInterrupt` in the third fit, and a real
    SIGINT during `method='de'`, both return the finished rows. The log level
    and signal handler are restored after an exception and after an interrupt.
19. The progress throttle, with a fake clock.

*Result, repair, persistence*

20. `table()`, `to_dataframe()`, `save_table()`, text and HTML agree. Zero and
    `None` uncertainties both show as missing.
21. `flagged()[16]` works, `flagged()[0]` raises, `flagged().update()` and
    `flagged().save_analysis()` raise.
22. `update()` replaces the row, marks it edited and recomputes flags;
    repairing the last flagged row empties `flagged()`. It refuses a fitter
    with a stale result, different data, or a different model, and leaves the
    row unchanged.
23. Save and load round-trip the table without touching data files. Save,
    move the directory with its data, reload, open a row; save again elsewhere
    and open a row.
24. A changed data file warns and keeps the row's numbers; `plot_fits()` draws
    a labelled theory curve for it. A custom model is refused unless allowed.
    A row document with a newer schema version is refused.
    `IQuana.load_analysis()` on a series file names the right call.
25. An interrupted run saves and reloads as "18 of 41 processed, interrupted".
26. The example script runs in `test_examples.py`.

## 9. Phase 2

Set aside for the next phase by the owner, and spelled out here to show that
phase 1 leaves room. None of this is built in phase 1.

### 9.1 2D datasets

*Use case:* a field or temperature scan of 2D patterns, the normal case for
polarised work on SKADI.

The loop is already dimension-agnostic, because `fit()` is. What is missing:

- **Replaying the pixel mask.** The template's mask operations (rings, boxes)
  must be applied to each dataset. `Selection.to_dict()` and
  `Selection.restore()` exist for the analysis file and do this; `_fit_one`
  gains one step after loading.
- **Magnetism** stays enabled across 2D datasets with no change.
- **Plots.** Table and trend plot are unchanged. `plot_fits()` for 2D shows
  one row at a time through the drill-down and `plotting2d`.
- **Persistence.** An in-memory 2D dataset is not embedded in an analysis file
  today. Until that is lifted, a 2D series from files saves completely, and a
  series of in-memory 2D datasets is refused by `save_analysis()`.
- **Metadata values** read each file twice (3.3). Measure that for 2D files
  and reuse the first read if it matters.

Then remove the phase 1 refusal of 2D data.

### 9.2 Parallel execution

*Use case:* hundreds of kinetic frames, or slow models.

```python
series = fitter.fit_series('frames/*.dat', start='template', workers=8)
```

- Only `start='template'` can run in parallel. `workers > 1` with
  `start='previous'` is an error that says why.
- **Mechanism:** `concurrent.futures.ProcessPoolExecutor`. A task is the
  template document, one item and the caller's `allow_custom_models`; the
  return value is one row. All of it is plain data already. A worker process
  builds its fitter with `fitter_from_document()`.
- **Order** of the result is the order of the items. Progress lines are
  printed as rows complete.
- **Kernel platform:** workers use the CPU kernel.
- **Interrupt:** cancel what has not started, keep the finished rows.
- **Windows** starts workers by spawning. A user script without
  `if __name__ == '__main__':` re-imports itself in every worker. The example
  script and the docs must show the guard.

The acceptance test is that `workers=4` gives rows identical to `workers=1`.

### 9.3 Bayesian sampling per dataset

*Use case:* uncertainties that can be trusted along a series.

```python
series = fitter.fit_series(files, bayesian=True, samples=10000)
```

- Each dataset is fitted with the optimiser as in phase 1, then sampled with
  `fit_bayesian()` from that optimum. Values and uncertainties are then the
  posterior ones, and the table header says so.
- **Chains are not kept.** A row stores the `PosteriorDigest` an analysis file
  already stores. For the chain of one row, run
  `series.fitter(i).fit_bayesian()`.
- **Trend plot** draws the 68% interval as the error bar, and labels it.
- **Flag:** `unmixed`, when the largest R-hat exceeds the threshold
  `fit_bayesian()` already reports.
- **Interrupt.** DREAM is reported to catch `KeyboardInterrupt` and return a
  truncated chain. The stop flag of 6.5 is passed to bumps as its abort test,
  and the row in progress is dropped, as in phase 1.
- **Cost** is N DREAM runs; the method logs an estimate after the first. In
  practice this wants 9.2.
- Composite models are refused, as `fit_bayesian()` refuses them today.

### 9.4 Later candidates, not designed

- `series.refit(rows='flagged', start='template')`: rerun those rows and keep
  the better fit. `start_row` is already stored, so no schema change.
- A flag for a jump in reduced χ² (Appendix B).
- Typed metadata beyond `sample.*`, once real facility files are known.
- A series held in one multi-entry file, through `load_data(dataset=...)`.
- Continuing an interrupted series where it stopped, as SASfit can.
- A series of model-free analyses (Guinier, Porod, invariant), once roadmap
  item 3.2 exists.
- Handing a finished series to `MultiFitter` as the start of a joint fit.
- Dividing the jump step by the spacing of the series variable.
- A display-only `unit=` for the variable. It would not change any key.

## 10. Order of work

1. Single-fit changes, each its own commit: lazy report; `_check_fit_setup()`
   with the free-parameter check; `_q_range_request` and its persistence;
   relaxed `_validated_param()`.
2. Refactors with no behaviour change: `_copy_setup()`,
   `fitter_from_document()`, shared envelope checks, log-level context
   manager.
3. `series.py`: items, values, order, loop, rows, flags, table and rendering.
4. Persistence, `fitter()`, `update()`.
5. Trend plot and `plot_fits()`.
6. Docs page, example script, notebook, changelog.

## 11. Open questions

1. **Decimal points in file names.** Some facilities write `T_20p5K`. The
   placeholder reads plain numbers only; such files need `values=`. Decide
   when real LOKI or SKADI file names are at hand.
2. **Metadata on real files.** Check one LOKI and one SKADI file for where
   temperature and field end up after sasdata reads them. That decides what
   the docs can promise for `values='sample.temperature'`.
3. **Relaxing the reader for out-of-bounds values** (section 5) changes
   `load_analysis()` for single fits from an error to a warning. The
   alternative is to store a clamped value in the configuration and the true
   one in the result, which keeps the reader strict but stores two numbers for
   one quantity. This document assumes the relaxed reader.
4. **`save_table` as the name** (3.7). The alternative is to keep
   `save_results` and document that it writes a different kind of file.

---

## Appendix A. What other tools do

### A.1 SasView

Facts from the SasView source (`main`, November 2026) and its fitting help.

- **Workflow.** Fit one dataset in a Fit Page, then send all datasets to a
  *Batch Page*, choose the same model, paste the parameters and press Fit.
  One model, one set of fit flags, one Q range and one smearing choice for
  every dataset.
- **Start values.** By default every fit starts from the page values. *Chain
  fitting* is an option that starts each fit from the previous result. The
  chain order is set by hand in a drag-and-drop list; there is no sorting by
  a measured quantity.
- **Result.** A grid with one row per dataset: reduced χ², the data name,
  every model parameter and an error column for each fitted one. There is no
  column for temperature or time and no status column. Export is CSV.
- **Trend plots.** SasView 4 could plot one grid column against another. That
  is not present in the current grid; the only plot is data plus fit for
  selected rows.
- **Execution.** Serial, in one background thread. Stopping a batch discards
  everything; there are no partial results.
- **Scripting.** No batch API. The documented route is a sasmodels example
  script that starts one `bumps` subprocess per file and collects the
  parameter values (no uncertainties, no χ²) into a CSV.
- **Open complaints in the issue tracker.** A failure in one dataset can take
  down the whole batch (#2749). The Q range cannot differ per dataset
  (#1819). The page does not show the fitted parameters of each dataset, and
  the curve it shows is the starting theory (#2018, #427). Saving a project
  stores the last row's parameters for all datasets (#2441). Fit curves can
  be plotted for only some rows (#3873). Earlier, batch and single fits
  disagreed because dI was ignored in batch (#1794, fixed).

### A.2 SASfit

Facts from the SASfit source (Tcl GUI code in the GitHub repository). The
online manual and the J. Appl. Cryst. papers could not be reached from the
environment this was written in, so nothing here comes from them.

- **Workflow.** *Options > run batch...* in the single-dataset fit window
  opens a "batch processing" window. The fit window must already be set up,
  so here too the user fits one curve first. Files are chosen as a directory
  plus a glob pattern, shown in a list and sorted in natural order. Files can
  be removed from the list but not reordered.
- **Start values.** Each file is loaded and the normal "Run fit" button is
  invoked, so every fit starts from whatever the previous one left behind.
  Chaining is implicit and cannot be switched off.
- **Running.** "Do all", or step through with "load next file". A run can be
  interrupted and continued.
- **Result.** One CSV per dataset holding the parameters with errors, moments
  of the size distribution, χ², reduced χ², R factors and the curves. There is
  no combined table of parameters against dataset for model fits, and no trend
  plot.
- **Series variable.** None. Only a run number and the file name are recorded.
- **Failures.** Nothing stops or marks a bad fit. An unreadable file raises a
  dialog and the previous result is saved again under the new name; a fit
  error is reported and the loop continues from the same parameters.
- **Model-free series.** The same file-list widget drives a series of
  Guinier/Porod analyses, and that one does append one row per file to a
  single CSV.
- **Global fit** is separate: all datasets are fitted at once, with parameters
  bound to shared global parameters.
- **Scripting.** None; the batch is driven through the GUI.

### A.3 What this design takes from them

| Lesson | Consequence here |
|---|---|
| "Fit one first, then apply to all" is the workflow people already know. | The configured `IQuana` is the template. |
| Chain fitting is essential for scans but its order is fragile. | Order comes from the series value or the file name, not from a hand-made list. An unusable fit is never used as a start. |
| A forced warm start with no failure handling (SASfit) lets one bad fit propagate unseen. | The warm start is the default but optional, and jumps are flagged. |
| A batch result without the series variable is only half a result. | The variable is a column of the table and the x axis of every trend plot. |
| One result file per dataset (SASfit) leaves the user to build the trend table. | One table, one file. |
| One bad dataset must not cost the other 999. | Every dataset is isolated; failures become flagged rows. An interrupt returns what is done. |
| Batch and single fits must give the same numbers. | The loop calls `IQuana.fit()` itself. |
| Users cannot see what happened to dataset *k*. | Any row opens as a normal `IQuana`. |
| Saving a batch loses per-dataset state. | Each row is stored as a complete single-fit analysis. |

## Appendix B. Review record

Three reviews of revision 1 (CC, GPT, DS) are folded into this revision.
Almost everything was adopted. The points below were not, or were adopted in a
different form.

| Proposal | Outcome | Reason |
|---|---|---|
| Stop the run after three identical failures (CC) | Not adopted | GPT and DS are right: three early frames outside the Q window can precede 997 good ones. Moving the call-wide checks into `_check_fit_setup()` removes the cause. |
| Run the jump test on reduced χ² (CC) | Deferred to 9.4 | Half of the rule (the uncertainty test) does not apply to χ², and the thresholds are untested even for parameters. Wait for real scans, as GPT argues. |
| Split into `series.py` and `series_results.py` now (CC) | Deferred | Nothing is lost by splitting when the module is written and its size is known (DS). Line-count estimates are removed (GPT). |
| `rebase_data_section()` helper (CC) | Different form | Rows carry no relative path until saved; the writer resolves and writes it (DS), using the load location first so a moved series re-saves correctly (GPT). |
| `flagged()` as a plain table (GPT) / as a view that writes through to its parent (DS) | Different form | A filtered `SeriesResult` keyed on original row numbers (DS), but read-only. That gives the reuse CC wanted and removes the ambiguity GPT objected to, without a parent reference. |
| `rows()`, `n_ok`, `n_flagged`, `n_failed` (DS edit list) | Not adopted | Iteration gives the rows and the header gives the counts. No use was shown for them. |
| An explicitly typed `qmax` equal to the data's end is written as open (DS §2.1 edge case) | Not adopted | Not needed: with the request stored, a typed end is kept as typed and an untyped end is `None`. The two cases are distinguishable. |
| Share `_copy_setup()` with `MultiFitter.add_fitter()` (revision 1) | Withdrawn | GPT and DS are right that the overlap is four lines and the responsibilities differ. |
| `series.get()` (revision 1; CC keep, DS drop) | Dropped | A fourth way to reach the same numbers, with unclear row selection. |
| Shorten phase 2 to use cases and constraints (GPT) | Not adopted | The owner asked for phase 2 to be spelled out in this document. |
