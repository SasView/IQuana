"""Series (batch) fitting: one configured fitter applied to many datasets.

``IQuana.fit_series()`` is the entry point. The configured fitter is the
template; every dataset is fitted by the public :meth:`IQuana.fit` on a fresh
copy of it, so a series of one file gives exactly the numbers of a single fit,
and everything the template knows (model, links, polydispersity, resolution,
weighting, Q cut) reaches every dataset without a second place to set it.

This module turns the call into an ordered list of datasets with their series
values, runs the fits and reports progress. What it returns, a
:class:`~iquana.series_results.SeriesResult`, is in :mod:`iquana.series_results`.
"""

import glob
import keyword
import logging
import os
import re
import signal
import threading
import time
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from .console import MIDDLE_DOT, OK, REDUCED_CHI_SQUARED, logger, raised_level
from .data.loader import load_sans_dataset
from .fitter import BOUND_WARNING, CONVERGENCE_WARNING, IQuana
from .persistence import build_analysis_dict
from .series_results import TABLE_COLUMNS, SeriesResult, SeriesRow, _finite, _number_text

__all__ = ['fit_series']

#: Seconds between progress lines while a series runs.
PROGRESS_INTERVAL = 10.0

_ORDERS = ('value', 'name')
_STARTS = ('previous', 'template')
_PLACEHOLDER = re.compile(r'\{([A-Za-z_]\w*)\}')
_PATTERN_TOKEN = re.compile(r'(\{[A-Za-z_]\w*\}|\*)')
_NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
#: Warnings a fit raises about conditions the series reports as flags.
_FLAGGED_WARNINGS = (BOUND_WARNING, CONVERGENCE_WARNING)
#: Read by the progress throttle; a test replaces it with a fake clock.
_clock = time.monotonic


# =========================================================================
# Which datasets, and their series value
# =========================================================================


@dataclass(frozen=True)
class _Item:
    label: str
    value: float
    #: An absolute path, or an in-memory dataset.
    source: Any


def _resolve_items(
    data: Any, values: Any, variable: str | None, order: str | None
) -> tuple[list[_Item], str]:
    """Turn the ``data``/``values``/``order`` arguments into ordered items and a variable."""
    if order is not None and order not in _ORDERS:
        raise ValueError(f"Unknown order {order!r}. Use 'value' or 'name'.")
    if isinstance(data, (str, os.PathLike)):
        text = os.fspath(data)
        if not os.path.isfile(text):
            if '*' not in text and not _PLACEHOLDER.search(text):
                raise ValueError(
                    f'No such file: {text}. A pattern needs * or a {{name}} placeholder.'
                )
            items, default = _pattern_items(text, values, order)
            return items, variable or default
        data = [text]
    if not isinstance(data, (list, tuple)):
        raise TypeError(
            'data must be a file path, a pattern, or a list of paths and datasets, '
            f'got {type(data).__name__}.'
        )
    items, default = _list_items(list(data), values, order)
    return items, variable or default


def _list_items(sources: list[Any], values: Any, order: str | None) -> tuple[list[_Item], str]:
    if not sources:
        raise ValueError('No datasets given: the list is empty.')
    if order is not None:
        raise ValueError('order= applies to a pattern; a list is fitted in the order given.')
    sources = [
        os.path.abspath(os.fspath(s)) if isinstance(s, (str, os.PathLike)) else s for s in sources
    ]
    labels = _labels(sources)
    if isinstance(values, str):
        numbers, name = _metadata_values(sources, labels, values)
    elif values is not None:
        numbers, name = _number_list(values, len(sources)), 'value'
    else:
        numbers, name = [float(i) for i in range(len(sources))], 'index'
    return [_Item(*entry) for entry in zip(labels, numbers, sources, strict=True)], name


def _pattern_items(pattern: str, values: Any, order: str | None) -> tuple[list[_Item], str]:
    matcher, finder, placeholder = _compile_pattern(pattern)
    paths = sorted(os.path.abspath(path) for path in glob.glob(finder) if os.path.isfile(path))
    if not paths:
        raise ValueError(f"No files match '{pattern}'.")
    if values is not None and not isinstance(values, str):
        raise ValueError(
            'A list of values goes with a list of datasets. A pattern is sorted, so '
            'the pairing of files and values would be in doubt: pass the files as a list.'
        )
    if placeholder is not None and values is not None:
        raise ValueError(
            f'Two sources for the series value: the {{{placeholder}}} placeholder and '
            f'values={values!r}. Use one.'
        )

    numbers: list[float] | None = None
    name = 'index'
    if placeholder is not None:
        numbers, name = [_number_in_name(matcher, path, placeholder) for path in paths], placeholder
    elif isinstance(values, str):
        numbers, name = _metadata_values(paths, [os.path.basename(p) for p in paths], values)

    order = order or ('value' if numbers is not None else 'name')
    if order == 'value' and numbers is None:
        raise ValueError(
            "order='value' needs a series value: a {name} placeholder in the pattern or "
            "values='<metadata path>'. Use order='name' for file-name order."
        )

    def name_key(path: str) -> tuple[Any, ...]:
        return _natural_key(os.path.basename(path)), _natural_key(path)

    if numbers is None:
        paths.sort(key=name_key)
        numbers = [float(i) for i in range(len(paths))]
    else:
        pairs = sorted(
            zip(paths, numbers, strict=True),
            key=(lambda pair: (pair[1], *name_key(pair[0])))
            if order == 'value'
            else (lambda pair: name_key(pair[0])),
        )
        paths, numbers = [p for p, _n in pairs], [n for _p, n in pairs]
    return [_Item(*entry) for entry in zip(_labels(paths), numbers, paths, strict=True)], name


def _compile_pattern(pattern: str) -> tuple[re.Pattern[str], str, str | None]:
    """A matcher, a glob and the placeholder name for *pattern*.

    Only ``*`` and one ``{name}`` are special; every other character, ``?`` and
    ``[`` included, is literal. ``*`` does not cross a directory separator, and
    ``**`` is the same as ``*`` (there is no recursive search). In the matcher
    ``*`` is lazy, so ``T_*{t}K`` reads 300 from ``T_ab300K``, not 0.
    """
    normalized = os.path.normpath(os.path.abspath(pattern))
    names = _PLACEHOLDER.findall(normalized)
    if len(names) > 1:
        raise ValueError(
            f'The pattern has {len(names)} placeholders ({", ".join(names)}); one at most.'
        )
    finder, matcher = [], []
    for token in _PATTERN_TOKEN.split(normalized):
        if token == '*':
            finder.append('*')
            matcher.append(r'[^\\/]*?')
        elif _PLACEHOLDER.fullmatch(token):
            finder.append('*')
            matcher.append(f'(?P<value>{_NUMBER})')
        elif token:
            finder.append(glob.escape(token))
            matcher.append(re.escape(os.path.normcase(token)))
    return re.compile(''.join(matcher)), ''.join(finder), (names[0] if names else None)


def _number_in_name(matcher: re.Pattern[str], path: str, placeholder: str) -> float:
    match = matcher.fullmatch(os.path.normcase(path))
    if match is None:
        raise ValueError(
            f'{os.path.basename(path)} matches the pattern but has no number where '
            f'{{{placeholder}}} is. Rename it, or remove it from the directory.'
        )
    value = _finite(match.group('value'))
    if value is None:
        raise ValueError(
            f'{os.path.basename(path)}: {match.group("value")!r} where {{{placeholder}}} is '
            'is not a finite number.'
        )
    return value


def _number_list(values: Any, count: int) -> list[float]:
    numbers = [_finite(value) for value in values]
    if len(numbers) != count:
        raise ValueError(f'values has {len(numbers)} entries for {count} datasets.')
    if None in numbers:
        raise ValueError(f'values must all be finite numbers, got {list(values)!r}.')
    return numbers


def _labels(sources: list[Any]) -> list[str]:
    """Paths relative to their common directory; in-memory datasets by title.

    Labels address rows, so they must be unique; a duplicate is an error.
    """
    paths = [os.path.dirname(source) for source in sources if isinstance(source, str)]
    try:
        base = os.path.commonpath(paths) if paths else None
    except ValueError:  # different drives on Windows
        base = None
    labels = []
    for position, source in enumerate(sources):
        if isinstance(source, str):
            labels.append(os.path.relpath(source, base) if base else source)
        else:
            labels.append(str(getattr(source, 'title', '') or f'dataset_{position}'))
    duplicates = sorted(label for label, n in Counter(labels).items() if n > 1)
    if duplicates:
        raise ValueError(
            f'Labels must be unique, but {", ".join(duplicates)} appear more than once. '
            'Give in-memory datasets distinct titles, or pass each file once.'
        )
    return labels


def _metadata_values(sources: list[Any], labels: list[str], path: str) -> tuple[list[float], str]:
    """Read the series value at the dotted *path* from every dataset, before any fit."""
    segments = path.split('.')
    if not all(segment.isidentifier() for segment in segments):
        raise ValueError(
            f"values={path!r} is not a dotted metadata path such as 'sample.temperature'."
        )
    numbers, units, unitless = [], set(), 0
    for source, label in zip(sources, labels, strict=True):
        try:
            dataset = load_sans_dataset(source).data if isinstance(source, str) else source
        except ValueError as exc:
            # Unlike a fit failure this cannot become a failed row: without its
            # value the file has no place in the series.
            raise ValueError(
                f'{exc}\n  Reading the series value needs every file. Remove the broken one, '
                'or pass the files as a list with values=[...].'
            ) from exc
        raw = _lookup(dataset, segments)
        number = _finite(_scalar(raw))
        if number is None:
            listing = ', '.join(_numeric_metadata(dataset)) or 'none'
            raise ValueError(
                f"{label} has no value at '{path}' (it is {raw!r}).\n"
                f'  Numeric metadata in this file: {listing}'
            )
        numbers.append(number)
        unit = _lookup(dataset, [*segments[:-1], f'{segments[-1]}_unit'])
        if isinstance(unit, str) and unit.strip():
            units.add(unit.strip())
        else:
            unitless += 1
    if len(units) > 1:
        raise ValueError(
            f"The datasets disagree on the unit of '{path}' ({', '.join(sorted(units))}). "
            'A series must use one unit: pass values=[...] in a common unit instead.'
        )
    if unitless:
        logger.info(
            f"Note: {unitless} of {len(sources)} datasets record no unit for '{path}', so "
            'its consistency across them was not checked.'
        )
    return numbers, segments[-1]


def _lookup(obj: Any, segments: Sequence[str]) -> Any:
    """Follow a dotted path: an attribute, or a key where the object is a dict."""
    for segment in segments:
        obj = obj.get(segment) if isinstance(obj, dict) else getattr(obj, segment, None)
        if obj is None:
            return None
    return obj


def _scalar(value: Any) -> Any:
    """A one-element array or list (as HDF5 readers deliver) as its element."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    size = getattr(value, 'size', None)
    if size == 1 and hasattr(value, 'item'):
        return value.item()
    return value


def _numeric_metadata(dataset: Any) -> list[str]:
    """``'sample.thickness = 1'``-style entries: the paths a user could pass instead."""
    found: list[str] = []

    def walk(obj: Any, path: str, depth: int) -> None:
        number = _finite(_scalar(obj))
        if number is not None:
            found.append(f'{path} = {number:g}')
            return
        if depth == 2 or obj is None or isinstance(obj, (str, bytes, list, tuple)):
            return
        if isinstance(obj, dict):
            entries = obj.items()
        elif hasattr(obj, '__dict__'):
            entries = ((n, getattr(obj, n, None)) for n in dir(obj) if not n.startswith('_'))
        else:
            return
        for name, value in entries:
            if isinstance(name, str) and name.isidentifier() and not callable(value):
                walk(value, f'{path}.{name}', depth + 1)

    for root in ('sample', 'meta_data'):
        walk(getattr(dataset, root, None), root, 0)
    return found


def _check_variable(name: str, parameters: Sequence[str], model_names: Sequence[str]) -> None:
    """*parameters* are the table's columns; *model_names* every model parameter.

    A model parameter is refused even when the template holds it fixed: a
    repaired row may vary it, and its name must never mean two things.
    """
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f'variable={name!r} must be a valid Python identifier: it names a column of '
            "the table. Units are not part of it ('temperature', not 'Temperature (K)')."
        )
    taken = set(TABLE_COLUMNS) | set(model_names) | set(parameters)
    taken |= {f'{p}_err' for p in parameters}
    if name in taken:
        raise ValueError(
            f"variable='{name}' collides with a model parameter or a column of the table. "
            'Pass variable= with another name.'
        )


def _duration_text(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f'{seconds} s'
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes} min {seconds} s' if minutes < 60 else f'{minutes // 60} h {minutes % 60} min'


def _natural_key(text: str) -> list[Any]:
    """``run2`` before ``run10``: digit runs compare as numbers."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', text)]


# =========================================================================
# The run
# =========================================================================


def fit_series(
    template: IQuana,
    data: Any,
    *,
    values: Any = None,
    variable: str | None = None,
    order: str | None = None,
    start: str = 'previous',
    engine: str = 'bumps',
    method: str | None = None,
    **kwargs: Any,
) -> SeriesResult:
    """Fit every dataset with the setup of *template*. See :meth:`IQuana.fit_series`."""
    if template.kernel is None:
        raise ValueError('No model set. Use set_model() on the template first.')
    if template.is_2d:
        raise ValueError(
            'fit_series() fits 1D data in this version; the template holds 2D data. '
            'A series of 1D cuts works: fitter.circular_average() or sector_average().'
        )
    if start not in _STARTS:
        raise ValueError(f"Unknown start {start!r}. Use 'previous' or 'template'.")
    template._check_fit_setup(engine, method)
    parameters = template._param_manager.free_parameter_names()

    items, variable = _resolve_items(data, values, variable, order)
    _check_variable(variable, parameters, list(template.params))
    # differential_evolution builds its population from the bounds: there is no
    # chain to follow, and start_row must not claim one.
    chain = start == 'previous' and method != 'differential_evolution'
    if start == 'previous' and not chain:
        logger.info(
            "Note: method='differential_evolution' starts from the bounds, not from the "
            "previous row. start='previous' has no effect; the rows are independent."
        )

    template_document = build_analysis_dict(template, None, include_result=False)
    template_document['data'] = None
    progress = _Progress(len(items), variable)
    logger.info(
        f'Series fit: {len(items)} datasets {MIDDLE_DOT} {template.model_name} {MIDDLE_DOT} '
        f'{len(parameters)} free parameters {MIDDLE_DOT} start={start}'
    )

    rows: list[SeriesRow] = []
    source, seed = template, None
    q_request = template._q_cut()
    interrupted = False
    with _interrupt_flag() as stop:
        try:
            for number, item in enumerate(items):
                worker = source._copy_setup()
                restarted = [] if seed is None else _leave_bounds(worker, template, seed)
                start_row = None if seed is None else seed.index
                row = _fit_one(worker, number, item, q_request, start_row, engine, method, kwargs)
                if stop.requested:  # the engine swallowed Ctrl-C: this fit was cut short
                    break
                if restarted:
                    note = (
                        f'{", ".join(restarted)} started from the template value: row '
                        f'{start_row} left {"it" if len(restarted) == 1 else "them"} on a bound.'
                    )
                    row = replace(row, warnings=(*row.warnings, note))
                rows.append(row)
                progress.report(row)
                if chain and row.usable_as_start:
                    source, seed = worker, row
        except KeyboardInterrupt:
            pass
        interrupted = len(rows) < len(items)

    series = SeriesResult(
        rows,
        variable=variable,
        template=template_document,
        start=start,
        engine=engine,
        method=method,
        fit_options=kwargs,
        planned=len(items),
        interrupted=interrupted,
        parameters=parameters,
        # The model came from this session's template, not from a file, so the
        # custom-model gate (which protects against untrusted files) is open.
        allow_custom_models=True,
    )
    progress.finish(series)
    if interrupted:
        warnings.warn(
            f'Series interrupted: {len(rows)} of {len(items)} datasets processed. The fit '
            'in progress was discarded; the result holds the finished rows.',
            stacklevel=3,
        )
    return series


def _fit_one(
    worker: IQuana,
    number: int,
    item: _Item,
    q_request: tuple[float | None, float | None] | None,
    start_row: int | None,
    engine: str,
    method: str | None,
    kwargs: dict[str, Any],
) -> SeriesRow:
    """Load, cut, fit and record one dataset. Any exception makes a failed row."""
    analysis, error = None, None
    with raised_level(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UserWarning)
        warnings.simplefilter('always', RuntimeWarning)
        try:
            if isinstance(item.source, str):
                worker.load_data(item.source)
            else:
                worker.set_data(item.source)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the series
            error = _message(exc)
        else:
            try:
                if worker.is_2d:
                    raise ValueError('2D data is not supported by fit_series() in this version.')
                if q_request is not None:
                    # Fails, and ends the row, when the cut leaves no points: falling
                    # through to a fit over the full range would look like a success.
                    worker.set_q_range(*q_request)
                worker.fit(engine, method, **kwargs)
            except Exception as exc:  # noqa: BLE001
                error = _message(exc)
            try:
                analysis = build_analysis_dict(worker, None)
            except Exception as exc:  # noqa: BLE001 - keep the first failure as the reason
                error = error or _message(exc)
    # Only the categories forced to 'always' above: library deprecation notices
    # that the active filters happen to let through are not about this dataset.
    texts = (
        str(record.message)
        for record in caught
        if issubclass(record.category, (UserWarning, RuntimeWarning))
    )
    kept = tuple(dict.fromkeys(t for t in texts if not t.startswith(_FLAGGED_WARNINGS)))
    return SeriesRow(
        index=number,
        label=item.label,
        value=item.value,
        source=item.source if isinstance(item.source, str) else None,
        analysis=analysis,
        error=error,
        warnings=kept,
        start_row=start_row,
    )


def _message(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


def _leave_bounds(worker: IQuana, template: IQuana, seed: SeriesRow) -> list[str]:
    """Start the parameters *seed* left on a bound from the template's values.

    Chaining carries the previous row's values. A start exactly on a bound is a
    trap for the default optimizer (bumps' amoeba simplex has no extent in that
    direction and never leaves the wall), so a scan that crosses a bound and
    comes back would stay pinned there. The template's value lies inside, and a
    parameter that really belongs on the bound (a background at zero) returns
    there in one fit. Returns the names that were moved.
    """
    moved = []
    for name, _side in seed.result.get('on_bounds') or []:
        if name in template.params:
            worker.set_param(name, value=template.params[name]['value'])
        elif name.endswith('_pd'):
            base = name[:-3]
            worker.set_pd_param(base, pd_width=template.get_pd_param(base)['pd'])
        else:
            continue
        moved.append(name)
    return moved


class _StopFlag:
    requested = False


@contextmanager
def _interrupt_flag() -> Iterator[_StopFlag]:
    """Notice Ctrl-C even when an engine swallows ``KeyboardInterrupt``.

    Some bumps methods catch the interrupt and return normally, which would
    record a fit cut short as a result and carry on with the next dataset. A
    SIGINT handler sets a flag first and then defers to the previous handler.
    Only in the main thread, where Python delivers signals; restored on exit.
    """
    flag = _StopFlag()
    if threading.current_thread() is not threading.main_thread():
        yield flag
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous is None:  # installed outside Python (an embedding host): leave it alone
        yield flag
        return

    def handler(signum: int, frame: Any) -> None:
        flag.requested = True
        if callable(previous):
            previous(signum, frame)

    signal.signal(signal.SIGINT, handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


class _Progress:
    """The run's log: the first row, every flagged or failed row, and a line every 10 s."""

    def __init__(self, total: int, variable: str):
        self.total = total
        self.variable = variable
        self.started = self.last = _clock()

    def report(self, row: SeriesRow) -> None:
        now = _clock()
        done = row.index + 1
        due = row.index == 0 or now - self.last >= PROGRESS_INTERVAL
        flags = row.own_flags()
        if not (due or flags or logger.isEnabledFor(logging.DEBUG)):
            return
        if row.failed:
            status = f'FAILED: {row.failure}'
        else:
            status = f'{REDUCED_CHI_SQUARED} {_number_text(row.reduced_chisq, ".3g")}'
            if flags:
                status += '   ' + '; '.join(f'{name}: {reason}' for name, reason in flags)
        line = f'  {"#" + str(row.index):>5}  {row.label}  {self.variable}={row.value:g}   {status}'
        if due and not flags:
            left = (now - self.started) / done * (self.total - done)
            line += f'   [{done}/{self.total}, ~{_duration_text(left)} left]'
        logger.info(line)
        self.last = now

    def finish(self, series: SeriesResult) -> None:
        counts = series._counts()
        hint = '   series.flagged() shows them.' if counts['flagged'] or counts['failed'] else ''
        done = (
            f'{len(series)} of {series.planned} datasets (interrupted)'
            if series.interrupted
            else f'{len(series)} datasets'
        )
        lines = [
            f'{OK} {done} in {_duration_text(_clock() - self.started)}: '
            f'{counts["ok"]} ok, {counts["flagged"]} flagged, {counts["failed"]} failed.{hint}'
        ]
        texts = series.warnings()
        if texts:
            rows = sum(bool(row.warnings) for row in series)
            lines.append(
                f'  warnings: {len(texts)} distinct ({rows} rows).   series.warnings() lists them.'
            )
        logger.info('\n'.join(lines))
