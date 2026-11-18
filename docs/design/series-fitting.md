# Design: batch (series) fitting

*Status: proposal, 2026-11-18, against IQuana 0.6.0. Target release: 0.8.*
*Implements item 3.1 of the roadmap ("Batch (series) fitting").*

## 1. Summary

Add one method, `IQuana.fit_series()`, and one result class, `SeriesResult`.

The user sets up and fits one curve exactly as today. The configured fitter is
then the template for the whole series:

```python
fitter = IQuana()
fitter.load_data('scan/T_020K.dat')
fitter.set_model('sphere')
fitter.set_param('radius', value=45, min=10, max=100)
fitter.set_param('scale', value=0.02, min=0.001, max=0.1)
fitter.fit()                                  # get one good fit first

series = fitter.fit_series('scan/T_{temperature}K.dat')

series                           # table: one row per file
series.plot('radius')            # radius against temperature
series.plot_fits()               # the curves with their fits
series.save_analysis('scan.json')
```

There is no new fitter class, no second place to set a model, and no new
parameter syntax. Everything the template knows (model, composites, structure
factor, polydispersity, links, bounds, resolution, weighting, Q range) applies
to every dataset because the loop reuses the single-fit code path unchanged.

The implementation is one new module of about 350 lines, one plotting module of
about 150 lines, and two small refactors of existing code (section 8).

**Decisions already taken** (with the project owner, 2026-11-18):

| Question | Decision |
|---|---|
| Entry point | `fitter.fit_series()` on a configured `IQuana`, not a `BatchFitter` class. |
| Series variable | A `{name}` placeholder in the file pattern, or an explicit `values=` list. Otherwise the file index. |
| Default start | `start='previous'`: each fit starts from the last successful one. `start='template'` is the alternative. |
| Phase 1 scope | 1D data, serial, optimiser fits, plus drill-down and repair of single rows. |
| Phase 2 scope | 2D datasets, parallel execution, Bayesian sampling per dataset (section 10). |

## 2. What other tools do

### 2.1 SasView

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
- **Trend plots.** SasView 4 could plot one grid column against another.
  That is not present in the current grid; the only plot is data plus fit
  for selected rows.
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

### 2.2 SASfit

*Research in progress; this section is not yet written.*

<!-- SASFIT-SECTION -->

### 2.3 What this design takes from them

| Lesson | Consequence here |
|---|---|
| "Fit one first, then apply to all" is the workflow people already know. | The configured `IQuana` is the template. No separate setup. |
| Chain fitting is essential for scans but its order is fragile. | Order is defined by the series variable, not by hand. A failed fit is never used as a start. |
| A batch result without the series variable is only half a result. | The variable is a column of the table and the x axis of every trend plot. |
| One bad dataset must not cost the other 999. | Every dataset is isolated; failures become flagged rows. An interrupt returns what is done. |
| Batch and single fits must give the same numbers. | The loop calls `IQuana.fit()` itself, so there is only one fit path. |
| Users cannot see what happened to dataset *k*. | Any row can be opened as a normal `IQuana`, with its data, setup and result. |
| Saving a batch loses per-dataset state. | Each row is stored as a complete single-fit analysis. |

## 3. Goals and non-goals

**Goals (phase 1)**

1. Fit N 1D datasets independently with the setup of one configured fitter.
2. Attach a series variable (temperature, time, concentration, field) to each
   dataset with as little typing as possible.
3. Return a table of parameters, uncertainties and χ² against that variable,
   and plot any parameter against it.
4. Flag datasets where the fit failed, did not converge, ended on a bound, or
   jumped away from its neighbours.
5. Save and reload the whole series as JSON; export the table as CSV.
6. Let the user open, repair and write back a single row.

**Non-goals (phase 1)**

- Shared or constrained parameters across datasets. That is `MultiFitter`.
- Different models, Q ranges or bounds per dataset. The drill-down (4.6) is
  the escape hatch for the few datasets that need special treatment.
- Fitting a trend (radius against temperature) to a function. The table gives
  the arrays; numpy and scipy do the rest.
- A progress bar widget, or any new dependency.
- 2D data, parallel execution and Bayesian sampling per dataset. These are
  phase 2 and are designed in section 10 so that phase 1 does not block them.

## 4. User interface

### 4.1 Signature

```python
def fit_series(
    self,
    data,                       # pattern, or list of paths / datasets
    values=None,                # one number per dataset
    variable=None,              # name of the series variable
    *,
    start='previous',           # or 'template'
    engine='bumps',
    method=None,
    **kwargs,                   # passed to fit(), as today
) -> SeriesResult: ...
```

`engine`, `method` and `**kwargs` mean exactly what they mean in `fit()`.

### 4.2 Saying which datasets, and what their series value is

Three forms cover the cases. All three end in the same internal list of
`(label, value, source)`.

**The value is in the file name.** A single `{name}` placeholder finds the
files, extracts the number, names the variable and sorts by it:

```python
series = fitter.fit_series('scan/T_{temperature}K.dat')
# finds T_020K.dat, T_022K.dat, ...  ->  temperature = 20, 22, ...
```

The placeholder matches a plain number (`20`, `-3.5`, `1e-3`). A file that
matches the pattern but has no number in that place is an error that names the
file; it is not skipped silently. More than one placeholder is an error.

**The value is known separately.** Give a list, and a name for it:

```python
series = fitter.fit_series(files, values=[0.5, 1, 2, 5], variable='concentration')
```

`data` may be a list of paths or of in-memory datasets (anything `set_data()`
accepts, such as `data_ops` results), or a mixture. `len(values)` must equal
`len(data)`.

**There is no value.** The x axis is the position in the series:

```python
series = fitter.fit_series('frames/*.dat')        # variable is 'index': 0, 1, 2 ...
```

**Order.** A pattern is sorted for you: by value when there is a placeholder,
otherwise in natural file-name order (`run_2` before `run_10`). An explicit
list is kept in the order given. This matters because `start='previous'`
follows the order.

`variable` may also be given with a pattern, to override the axis label
(`variable='Temperature (K)'`).

### 4.3 Start values

| `start=` | Each fit starts from | Use when |
|---|---|---|
| `'previous'` (default) | The fitted values of the last successful dataset. The first starts from the template. | Parameters drift along the series: temperature scans, kinetics. |
| `'template'` | The template's current values, every time. | Datasets are unrelated, or to cross-check a chained run. Results do not depend on order. |

Only values are carried over. Bounds and vary flags always come from the
template.

A row flagged `failed` or `unconverged` is never used as a start; the next fit
starts from the last good row (or the template if there is none). So one bad
dataset cannot poison the rest of a chain.

A chain can still be led astray by a fit that is wrong but looks successful.
The `jump` and `bound` flags exist to show that, and re-running with
`start='template'` is the check. The docs will say so.

### 4.4 What the user sees while it runs

The per-fit reports are silenced. One line per dataset is logged instead:

```text
Series fit: 41 datasets · sphere · 3 free parameters · start=previous
  [ 1/41] T_020K.dat   temperature=20    χ²ᵣ 1.04
  [ 2/41] T_022K.dat   temperature=22    χ²ᵣ 1.10
  ...
  [17/41] T_052K.dat   temperature=52    χ²ᵣ 9.83   bound: radius (max)
  [18/41] T_054K.dat   temperature=54    FAILED: No data points in Q range [0.01, 0.3]
  ...
✓ Series done in 38 s: 38 ok, 2 flagged, 1 failed.  series.flagged() lists them.
```

`set_verbosity('quiet')` removes the lines, as it does for everything else.
Warnings raised by a fit are collected on its row and not printed one by one;
a thousand-frame run would otherwise bury the screen.

**Ctrl-C** stops the loop and returns the rows finished so far, with a warning
that says how many were not reached. SasView loses everything in this case.

### 4.5 The result: `SeriesResult`

In a notebook, `series` renders as a table. `print(series)` gives the same as
text:

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

Columns are the label, the series variable, every **varied** parameter with
its uncertainty, reduced χ² and the flags. Fixed parameters are the same in
every row and stay in the template.

| Member | Returns |
|---|---|
| `len(series)`, `series[i]` | Number of rows; one `SeriesRow` (by position or by label). |
| `series.variable` | Name of the series variable. |
| `series.table()` | `list[dict]`, one flat dict per row: `label`, the variable, `radius`, `radius_err`, ..., `chisq`, `reduced_chisq`, `n_points`, `flags`. |
| `series.to_dataframe()` | The same as a pandas DataFrame. pandas is imported here only; it stays an optional dependency. |
| `series.get('radius')` | `(x, value, stderr)` numpy arrays over the successful rows, for the user's own trend analysis. |
| `series.flagged()` | The rows carrying any flag, as a shorter table with the reason spelled out. |
| `series.plot(*parameters)` | Trend plot (4.5.1). |
| `series.plot_fits(rows=None)` | Data and fit curves (4.5.2). |
| `series.fitter(i)` | A normal `IQuana` for row *i* (4.6). |
| `series.update(i, fitter)` | Replace row *i* with a repaired fit (4.6). |
| `series.save_results('scan.csv')` | The table as CSV. Same name and meaning as `IQuana.save_results()`. |
| `series.save_analysis('scan.json')` | Everything, reloadable (section 6). |
| `SeriesResult.load_analysis('scan.json')` | The reverse. |

#### 4.5.1 Trend plot

`series.plot()` with no arguments draws every varied parameter, and reduced
χ² at the bottom, as stacked panels over a shared x axis:

```text
 radius (Å)    |            o
           60  |         o     ◇        o = fitted value with error bar
           50  | o  o  o                ◇ = flagged row (hover shows why)
               +---------------------
 scale         | o  o  o  o
               |            ◇
               +---------------------
 χ²ᵣ           |            ◇
               | o  o  o  o
               +---------------------
                 20  30  40  50  60   temperature
```

`series.plot('radius')` or `series.plot('radius', 'scale')` selects panels.
Flagged rows use a different marker and say why on hover. Failed rows have no
value to draw; they are counted in the title. The figure is a plotly figure,
returned and shown under the same rules as every other IQuana plot.

#### 4.5.2 Curves and fits

`series.plot_fits()` overlays data and fit for up to nine rows spread evenly
through the series, coloured by the series value. `rows=[3, 16, 17]` or
`rows='flagged'` chooses others. It reuses `multi_plotting.plot_multi`.

### 4.6 Drill down and repair

A flagged row needs a person. The row opens as an ordinary fitter with its
data, setup and result in place:

```python
series.flagged()                 # row 16: bound (radius at max), jump
bad = series.fitter(16)          # a normal IQuana
bad.plot_results()               # look at it
bad.set_param('radius', value=60, max=200)
bad.fit()
series.update(16, bad)           # write the repaired fit back
```

`update()` refuses a fitter whose data is not the row's data (compared by the
array fingerprint IQuana already records), so a row cannot be overwritten by
the wrong dataset. The row is marked `edited` in the table and in the file, so
a reader can tell that it did not come out of the automatic run. Flags are
recomputed for the whole series afterwards, because `jump` depends on the
neighbours.

`series.fitter(i)` is also the hand-over path: `series.fitter(i).save_analysis(...)`
or `.report(...)` gives a visiting user one dataset as a self-contained file.

## 5. Flags

A flag means "look at this row". It never removes a row or changes a number.

| Flag | Condition | Source |
|---|---|---|
| `failed` | Loading or fitting raised. The row has no result; the message is kept. | New: the loop catches `Exception` per dataset. |
| `unconverged` | The engine reported `converged is False`. | Existing `FitResultContract.converged`. bumps reports `None`, so in practice this is the lmfit engine. |
| `bound` | A varied parameter ended on a bound. | Existing `FitResultContract.on_bounds`. |
| `jump` | A varied parameter moved much more than is usual in this series. | New, below. |
| `edited` | The row was replaced with `update()`. | New. |

**The jump rule.** For each varied parameter, take the absolute steps between
consecutive successful rows. Row *i* is flagged when its step from row *i−1*
is both

- more than 5 times the median step of that parameter in the series, and
- more than 3 times the combined uncertainty of the two values (this second
  test is skipped when an uncertainty is not available).

It needs at least five successful rows; below that, nothing is flagged. The
two thresholds are module constants, not arguments.

It is computed after the loop, from the table alone, so it is the same for
both start modes and after a reload. It will flag a real phase transition.
That is intended: a transition is exactly where a person should look at the
fit. It does not account for uneven spacing of the series variable; if that
proves to matter in practice, the step can be divided by the spacing later.

High reduced χ² is not a flag. It is a column and a plot panel, and what
counts as "high" depends on the error bars of the instrument.

## 6. Persistence

"Save and reload like a single analysis" is met by storing each row **as** a
single analysis.

`save_analysis()` on an `IQuana` already writes a JSON document holding the
data reference (path, relative path, SHA-256, or the arrays of an in-memory
dataset), the full configuration, resolution, weighting, Q range and the
result. A series file is a thin wrapper around N of these:

```json
{
  "schema": {"format": "iquana-series", "version": 1, "written_by": "0.8.0"},
  "variable": "temperature",
  "start": "previous",
  "engine": "bumps",
  "method": "amoeba",
  "template": { "...": "single-analysis document, without data or result" },
  "rows": [
    {
      "label": "T_020K.dat",
      "value": 20.0,
      "flags": [],
      "warnings": [],
      "error": null,
      "analysis": { "...": "single-analysis document, schema iquana-analysis" }
    }
  ]
}
```

Consequences:

- **No new result schema.** `rows[i].analysis` is produced by the existing
  `persistence.build_analysis_dict()` and read by the existing reader.
  Every guarantee of the single-fit file (number encoding, the check that a
  result still describes its setup, the warning when a data file has changed
  or moved, the refusal of custom models unless allowed) holds per row
  without new code.
- **Any row is a valid analysis on its own.** `series.fitter(i)` is "load this
  document". It works the same in the session that ran the fit and after a
  reload, so there is one drill-down path, not two.
- **Failed rows** have `analysis.result = null` and an `error` string. Their
  data reference is still stored, so the row can be opened and repaired.
- **Data files are referenced, not copied**, exactly as today. In-memory
  datasets are embedded, exactly as today.
- **Size.** A row is a few kilobytes (configuration plus a small covariance
  matrix). A thousand rows give a file of a few megabytes. Fitted curves are
  not stored; they are recomputed on demand, as for a single analysis.

`SeriesResult.load_analysis()` takes the same `allow_custom_models` argument
as `IQuana.load_analysis()` and passes it through.

## 7. How it works

### 7.1 The loop

```python
def fit_series(template, data, values, variable, *, start, engine, method, **kwargs):
    items, variable = _resolve_items(data, values, variable)   # [(label, value, source)]
    _validate(template, start)                                 # fail before fitting anything

    worker = template._copy_setup()            # shares the compiled kernel, copies the rest
    next_start = worker._param_manager         # the parameter state the next fit starts from
    q_range = template._restricted_q_range()   # None when the template uses the full range

    rows = []
    try:
        for label, value, source in items:
            worker._param_manager = copy.deepcopy(next_start)
            row = _fit_one(worker, label, value, source, q_range, engine, method, kwargs)
            rows.append(row)
            if start == 'previous' and row.usable_as_start:   # not failed, not unconverged
                next_start = worker._param_manager            # holds the fitted values
    except KeyboardInterrupt:
        warnings.warn(f'Series interrupted: {len(rows)} of {len(items)} datasets fitted.')
    return SeriesResult(rows, variable, ...)     # computes the jump flags
```

The two start modes differ in one line: whether `next_start` ever moves.

`_fit_one` is the whole per-dataset procedure:

1. `worker.load_data(path)` or `worker.set_data(obj)`.
2. Re-apply the template's Q range. This step is needed because the Q range
   lives on the dataset and is reset by `load_data()`. If the dataset has no
   points in the range, `set_q_range()` raises and the row fails with that
   message.
3. `worker.fit(engine, method, **kwargs)` inside `warnings.catch_warnings`.
4. Build the row: label, value, the result contract, the collected warnings,
   and the row's analysis document from `build_analysis_dict(worker, ...)`.

Any `Exception` in steps 1 to 3 becomes a `failed` row. `KeyboardInterrupt`
is not an `Exception` and passes through to the outer handler.

Because step 3 is the public `fit()`, every pre-fit check that exists today
(start outside bounds, zero dI, scale degeneracy, model against data) runs
per dataset, and a batch fit of one file gives the same numbers as a single
fit of it. This is tested (section 9).

### 7.2 The template is not modified

`fit_series()` works on a copy. After it returns, the template has the same
data, parameters and result as before. A user can run it twice with different
`start=` and compare.

The template needs a model. It does not need data or a fit of its own, though
fitting one dataset first is the workflow the docs will teach.

### 7.3 Up-front validation

Before the first fit, raise `ValueError` for: no model set; no dataset
matched; `values` of the wrong length; an unknown `start`; a 2D template or
dataset (phase 1); a template whose start values are outside its bounds (the
existing `_check_start_values()`). A mistake in the call should cost
no fits, not a full run of failed rows.

### 7.4 Data classes

```python
@dataclass
class SeriesRow:
    label: str
    value: float
    flags: tuple[str, ...]
    error: str | None              # set when the row failed
    warnings: tuple[str, ...]
    analysis: dict                 # single-analysis document (section 6)
    parameters: dict               # name -> {'value', 'stderr'} for varied parameters
    chisq: float | None
    reduced_chisq: float | None
    n_points: int | None
```

`parameters` and the χ² fields are read out of the result once, so the table,
the plots and the flags do not parse documents repeatedly. `SeriesResult`
holds the rows, `variable`, `start`, `engine`, `method` and the template
document. Neither class holds a fitter, a kernel or a dataset, so a
`SeriesResult` of a thousand rows is small and can be deep-copied and
serialised without special cases.

## 8. Changes to the code

**New**

| File | Contents | Size |
|---|---|---|
| `src/iquana/series.py` | `fit_series()`, `_resolve_items()`, `_fit_one()`, flags, `SeriesRow`, `SeriesResult` with table, text and HTML rendering, CSV, save and load. | about 350 lines |
| `src/iquana/series_plotting.py` | Trend plot. `plot_fits` is a thin call into `multi_plotting.plot_multi`. | about 150 lines |
| `tests/test_series.py` | Section 9. | |
| `docs/series.md`, `examples/series_fitting_example.py`, `notebooks/series_fitting.ipynb` | User guide, script, notebook. | |

**Changed**

| File | Change | Why |
|---|---|---|
| `fitter.py` | Add `fit_series()`, a thin method that delegates to `series.fit_series`, as `report()` delegates to `reporting`. | Entry point. |
| `fitter.py` | Add private `_copy_setup()`: a new `IQuana` sharing the kernel, with deep copies of the parameter manager, resolution and weighting, and no data. | DRY: `MultiFitter.add_fitter()` already does this inline; it will call the helper. |
| `fitter.py` | Add private `_restricted_q_range()`: the Q range if the user narrowed it, else `None`. | The loop must not "restrict" each dataset to the template dataset's full range. |
| `persistence.py` | Split `read_analysis(filename)` into reading the file and a new `fitter_from_document(document, base_path, data=None)`. | DRY: the drill-down loads a row from a dict, not from a file. No behaviour change for `load_analysis()`. |
| `console.py` | A small context manager that raises the log level for a block and restores it. | Silencing per-fit reports without touching the user's `set_verbosity` choice. |
| `__init__.py` | Export `SeriesResult`. | |
| `docs/multifit.md`, `README.md`, `CHANGELOG.md`, `mkdocs.yml` | Replace "batch fitting is not supported" with a pointer; add the page to the nav. | |

Nothing changes in the engines, the parameter manager, the result contract or
the analysis schema.

**Dependencies:** none added.

## 9. Tests

All on simulated data from `examples.simulate()`, with a known parameter ramp
(for example a sphere radius rising linearly over 12 datasets).

1. **Truth recovery.** The fitted radius follows the ramp within its
   uncertainty, for both start modes and both engines.
2. **Same numbers as a single fit.** A series of one file equals
   `IQuana.fit()` on that file, value for value.
3. **Template untouched.** Parameters, data and result of the template are
   identical before and after.
4. **Setup carries over.** Q range, resolution, weighting, polydispersity, a
   composite model and a structure factor each reach every row.
5. **Items.** Placeholder extraction and sorting; natural sort without a
   placeholder; list order kept; length mismatch, no match, two placeholders
   and a non-numeric match each raise before any fit.
6. **Isolation.** One unreadable file and one dataset outside the Q range in
   the middle of a series: those rows are `failed`, the others are fitted, and
   under `start='previous'` the row after a failure starts from the last good
   one.
7. **Flags.** A series built to hit a bound, and one built with a
   discontinuity, are flagged at the right rows; a smooth ramp has no flags;
   fewer than five rows never flag `jump`.
8. **Interrupt.** A `KeyboardInterrupt` raised in the third fit returns two
   rows and a warning.
9. **Persistence.** Save and load round-trips the table exactly; a reloaded
   row opens with `fitter(i)` and plots; a moved or changed data file warns as
   a single analysis does; a custom model is refused unless allowed.
10. **Repair.** `update()` replaces the row, marks it `edited`, recomputes
    flags, and refuses a fitter holding different data.
11. **Output.** `table()`, `to_dataframe()`, `save_results()` and the text and
    HTML renderings agree with each other.
12. The example script runs in `test_examples.py`, as the others do.

## 10. Phase 2

These three were set aside for the next phase. Each is sketched here to show
that phase 1 leaves room for it; none of this is built in phase 1.

### 10.1 2D datasets

*Use case:* a field or temperature scan of 2D patterns, the normal case for
polarised work on SKADI.

The loop is already dimension-agnostic, because `fit()` is. What is missing:

- **Replaying the pixel mask.** The template's mask operations (rings, boxes)
  must be applied to each dataset. `Selection.to_dict()` and
  `Selection.restore()` exist for the analysis file and do exactly this; the
  per-dataset procedure gains one step after loading.
- **Magnetism** stays enabled across 2D datasets with no change.
- **Plots.** The table and trend plot are unchanged. `plot_fits()` for 2D
  shows one row at a time through the drill-down and `plotting2d`.
- **Persistence.** An in-memory 2D dataset is not embedded in an analysis
  file today. Until that limit is lifted, a 2D series loaded from files saves
  completely and a series of in-memory 2D datasets is refused by
  `save_analysis()` with a message saying so.

Then remove the phase 1 refusal of 2D data. Estimated at a few days,
most of it tests.

### 10.2 Parallel execution

*Use case:* hundreds of kinetic frames, or slow models (polydisperse,
structure factor, 2D).

```python
series = fitter.fit_series('frames/*.dat', start='template', workers=8)
```

- Only `start='template'` can run in parallel, because the fits are then
  independent. `workers > 1` with `start='previous'` is an error that says
  why. No chunked or approximate chaining.
- **Mechanism:** `concurrent.futures.ProcessPoolExecutor`. A task is the
  template document plus one data reference; the return value is one row.
  Both are already plain JSON-able data (sections 6 and 7.4), so nothing new
  has to be made picklable. A worker process rebuilds its fitter with
  `fitter_from_document()`, the same function the drill-down uses, once per
  process.
- **Order** of the result is the order of the items, whatever order the
  workers finish in. Progress lines are printed as rows complete.
- **Kernel platform:** workers use the CPU kernel. Several processes sharing
  one GPU through OpenCL is not something to discover in production.
- **Interrupt:** cancel what has not started, keep the finished rows.

The acceptance test is that `workers=4` gives rows identical to `workers=1`.

### 10.3 Bayesian sampling per dataset

*Use case:* uncertainties that can be trusted along a series, especially for
correlated parameters.

```python
series = fitter.fit_series(files, bayesian=True, samples=10000)
```

- Each dataset is first fitted with the optimiser as in phase 1, then sampled
  with `fit_bayesian()` from that optimum. The row's values and uncertainties
  are then the posterior ones, and the table header says so.
- **Chains are not kept.** A row stores the posterior digest that an analysis
  file already stores (`PosteriorDigest`: the statistics, not the samples).
  A thousand chains in memory is not acceptable, and the user who wants the
  chain of one row runs `series.fitter(i).fit_bayesian()`.
- **Trend plot** draws the 68% interval as the error bar, and labels it.
- **Flag:** a new `unmixed` flag when the largest R-hat of a row exceeds the
  threshold `fit_bayesian()` already reports.
- **Cost** is N times a DREAM run. The method logs an estimate after the first
  dataset. In practice this feature wants 10.2.
- Composite models are refused, as `fit_bayesian()` refuses them today.

### 10.4 Later candidates, not designed

- A series held in one file (many entries in one NXcanSAS file), through the
  `dataset=` selector that `load_data()` already has.
- Handing a finished series to `MultiFitter` as a starting point for a joint
  fit with shared parameters.
- Dividing the jump step by the spacing of the series variable.

## 11. Order of work

1. Refactors: `_copy_setup()`, `_restricted_q_range()`, `fitter_from_document()`,
   log-level context manager. No behaviour change; existing tests must pass
   untouched.
2. `series.py`: items, loop, rows, table, text and HTML rendering. Tests 1 to 8, 11.
3. Persistence and drill-down. Tests 9, 10.
4. Trend plot and `plot_fits`.
5. Docs page, example script, notebook, changelog.

Steps 1 to 3 are the feature. Each step is a reviewable pull request.

## 12. Open questions

1. **Decimal points in file names.** Some facilities write `T_20p5K`. The
   placeholder reads plain numbers only; such files need `values=`. Worth
   supporting only if real file names from LOKI or SKADI look like this.
2. **Units of the variable.** The variable is a bare name. A unit can be put
   in the label (`variable='Temperature (K)'`). A separate `unit=` argument
   would be cleaner in the table header but is one more argument.
3. **Default of `plot_fits()`.** Nine evenly spaced rows is a guess. The
   alternative default is "all flagged rows".
