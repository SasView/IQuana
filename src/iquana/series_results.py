"""The result of a series fit: :class:`SeriesResult` and its rows.

A :class:`SeriesResult` is a table of fitted parameters against a series
variable, with each row stored as a complete single-fit analysis document. That
one decision carries most of this module: a row opens as a normal ``IQuana``
(:meth:`SeriesResult.fitter`), saves and reloads through the existing analysis
reader, and needs no result schema of its own.

Flags mark the rows a person should look at. They are never stored: they are
recomputed from the rows on construction, on load and after every
:meth:`SeriesResult.update`, so a file cannot carry flags that contradict its
numbers.

The run that produces a result is in :mod:`iquana.series`.
"""

import copy
import csv
import io
import itertools
import json
import math
import os
import statistics
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

from . import __version__, series_plotting
from .console import MIDDLE_DOT, OK, PLUS_MINUS, REDUCED_CHI_SQUARED, logger
from .fileio import atomic_write
from .fitter import IQuana
from .persistence import (
    SERIES_FORMAT,
    AnalysisFileError,
    _require,
    build_analysis_dict,
    check_analysis_document,
    check_schema,
    decode_numbers,
    fitter_from_document,
    locate_data_file,
    read_json,
    relocate_data_section,
)
from .report import escape_html_cell

__all__ = ['SeriesResult', 'SeriesRow']

#: Version of the series file format (the envelope; rows are analysis documents).
SERIES_VERSION = 1

#: A row is flagged ``jump`` when its step exceeds this multiple of the median
#: step of that parameter over the series...
JUMP_MEDIAN_FACTOR = 5.0
#: ...and this multiple of the combined uncertainty of the two values. Both are
#: a first guess, not yet tried on real scans.
JUMP_SIGMA_FACTOR = 3.0
#: Fewer steps than this between converged neighbours, and nothing is flagged.
JUMP_MIN_STEPS = 5

#: Rows shown by the text and HTML tables before they are abbreviated.
DISPLAY_ROWS = 20
#: Rows ``plot_fits()`` draws by default, spread evenly over the series.
DEFAULT_FIT_PLOTS = 9

#: Fixed columns of :meth:`SeriesResult.table`; the series variable may not take them.
TABLE_COLUMNS = ('row', 'label', 'chisq', 'reduced_chisq', 'n_points', 'flags', 'edited', 'error')


# =========================================================================
# Rows
# =========================================================================


# eq=False: rows compare and hash by identity (the analysis dict is unhashable).
# No __slots__: cached_property needs the instance __dict__, frozen or not.
@dataclass(frozen=True, repr=False, eq=False)
class SeriesRow:
    """One dataset of a series: what was fitted, and how it went.

    The fields up to ``source_relative`` are what a series file stores.
    Parameter values, χ² and the point count are read out of ``analysis`` (a
    single-analysis document) on demand. ``flags`` and ``variable`` are filled
    in by the owning :class:`SeriesResult` and are never stored.
    """

    #: Row number in the original run. It never changes, so it is an identity.
    index: int
    label: str
    value: float
    #: The path that was attempted; None for an in-memory dataset.
    source: str | None
    #: The single-analysis document; None when the dataset never loaded.
    analysis: dict[str, Any] | None
    #: Why the row failed, or None.
    error: str | None
    warnings: tuple[str, ...] = ()
    #: The row whose fitted values seeded this one, or None for the template.
    start_row: int | None = None
    #: Replaced by :meth:`SeriesResult.update` after the run.
    edited: bool = False
    #: ``source`` relative to the series file it was read from, so a row whose
    #: file never loaded can still be retried after the project moved.
    source_relative: str | None = None
    #: ``(flag, reason)`` pairs, set by the owning series.
    flags: tuple[tuple[str, str], ...] = ()
    #: Name of the series variable, for the row's own display.
    variable: str = 'value'

    @cached_property
    def result(self) -> dict[str, Any] | None:
        """The stored fit result, or None for a row without one."""
        return None if self.analysis is None else self.analysis.get('result')

    @property
    def failed(self) -> bool:
        return self.error is not None or self.result is None

    @property
    def failure(self) -> str | None:
        """Why the row has no result, or None when it has one."""
        return (self.error or 'no fit result was stored') if self.failed else None

    @property
    def flag_names(self) -> tuple[str, ...]:
        return tuple(name for name, _reason in self.flags)

    @cached_property
    def _parameters(self) -> dict[str, dict[str, Any]]:
        return dict((self.result or {}).get('parameters') or {})

    @cached_property
    def _configuration(self) -> dict[str, Any]:
        return decode_numbers((self.analysis or {}).get('configuration') or {})

    @cached_property
    def _bounds_table(self) -> dict[str, Any]:
        return self._configuration.get('params') or {}

    @cached_property
    def on_bounds(self) -> frozenset[str]:
        """The varied parameters this row's fit left on a bound."""
        return frozenset(name for name, _side in (self.result or {}).get('on_bounds') or [])

    def varied(self) -> list[str]:
        """The parameters this row's fit varied, in the model's parameter order.

        A polydispersity width follows its parameter (``radius``, ``radius_pd``).
        """
        position = {name: i for i, name in enumerate(self._bounds_table)}

        def key(name: str) -> tuple[int, bool]:
            width = name.endswith('_pd') and name not in position
            return position.get(name[:-3] if width else name, len(position)), width

        names = [name for name, info in self._parameters.items() if not info.get('fixed', True)]
        return sorted(names, key=key)

    def estimate(self, name: str) -> tuple[float | None, float | None]:
        """``(value, stderr)`` of *name*, each None when missing or not finite.

        A zero uncertainty is None too, meaning "not available", as in the fit
        report: it is what a fixed parameter reports.
        """
        info = self._parameters.get(name)
        if info is None and name.endswith('_pd'):
            # A width a repair held fixed is not in the result: read the setting.
            widths = (self._configuration.get('polydispersity') or {}).get('params') or {}
            return _finite((widths.get(name[:-3]) or {}).get('pd')), None
        info = info or {}
        return _finite(info.get('value')), _finite(info.get('stderr'), positive=True)

    def is_fixed(self, name: str) -> bool:
        return bool((self._parameters.get(name) or {}).get('fixed', True))

    def bounds(self, name: str) -> tuple[float, float]:
        """The bounds *name* was fitted within. Polydispersity widths are [0, 1]."""
        info = self._bounds_table.get(name)
        if info is not None:
            return float(info['min']), float(info['max'])
        if name.endswith('_pd'):
            return 0.0, 1.0
        return -math.inf, math.inf

    @property
    def chisq(self) -> float | None:
        return _finite((self.result or {}).get('chisq'))

    @property
    def reduced_chisq(self) -> float | None:
        return _finite((self.result or {}).get('reduced_chisq'))

    @property
    def n_points(self) -> int | None:
        return None if self.result is None else self.result.get('n_points')

    @property
    def converged_fit(self) -> bool:
        """A completed fit the engine did not call unconverged."""
        return not self.failed and self.result.get('converged') is not False

    def outside_bounds(self) -> list[str]:
        """Varied values that ended outside their bounds (``leastsq`` ignores them)."""
        found = []
        for name in self.varied():
            value, _stderr = self.estimate(name)
            low, high = self.bounds(name)
            if value is not None and not low <= value <= high:
                found.append(f'{name} = {value:g} outside [{low:g}, {high:g}]')
        return found

    @property
    def usable_as_start(self) -> bool:
        """Whether the next fit of a chain may start from this row's values.

        The fit completed, the engine did not report non-convergence, and every
        varied value is finite and inside its bounds. A value resting **on** a
        bound is usable: a background that really goes to zero should chain.
        """
        if not self.converged_fit:
            return False
        values = [self.estimate(name)[0] for name in self.varied()]
        return all(value is not None for value in values) and not self.outside_bounds()

    def own_flags(self) -> list[tuple[str, str]]:
        """The flags this row earns on its own; ``jump`` needs its neighbours."""
        if self.failed:
            return [('failed', self.failure)]
        flags = []
        if self.result.get('converged') is False:
            flags.append(('unconverged', f'did not converge: {self.result.get("message") or "-"}'))
        outside = self.outside_bounds()
        if outside:
            flags.append(('out_of_bounds', '; '.join(outside)))
        hits = self.result.get('on_bounds') or []
        if hits:  # in the result's order, with the side each one hit
            flags.append(('bound', '; '.join(self._bound_text(name, side) for name, side in hits)))
        return flags

    def _bound_text(self, name: str, side: str) -> str:
        low, high = self.bounds(name)
        edge, limit = ('lower', low) if side == 'min' else ('upper', high)
        return f'{name} at {edge} bound ({limit:g})'

    def why(self) -> str:
        return '; '.join(
            f'{name}: {reason}' if name == 'failed' else reason for name, reason in self.flags
        )

    def __repr__(self) -> str:
        lines = [
            f'Row {self.index} {MIDDLE_DOT} {self.label} {MIDDLE_DOT} '
            f'{self.variable} = {self.value:g}',
            '  started from: '
            + (
                'the template'
                if self.start_row is None
                else f'row {self.start_row} in the original run'
            ),
        ]
        if self.failed:
            lines.append(f'  FAILED: {self.failure}')
        else:
            hits = dict(self.result.get('on_bounds') or [])
            width = max((len(name) for name in self.varied()), default=0)
            for name in self.varied():
                note = f'   {self._bound_text(name, hits[name])}' if name in hits else ''
                lines.append(f'  {name:<{width}}  {_estimate_text(self, name)}{note}')
            lines.append(
                f'  {REDUCED_CHI_SQUARED} {_number_text(self.reduced_chisq)} {MIDDLE_DOT} '
                f'{self.n_points} points {MIDDLE_DOT} flags: '
                + (', '.join(self.flag_names) or '(none)')
                + (' (edited)' if self.edited else '')
            )
        lines.append('  warnings: ' + ('; '.join(self.warnings) or '(none)'))
        lines.append(f'  -> series.fitter({self.index}) opens it as a normal IQuana')
        return '\n'.join(lines)


def _finite(value: Any, *, positive: bool = False) -> float | None:
    """*value* as a float when it is a finite number (and > 0 if *positive*)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


# =========================================================================
# Flags
# =========================================================================


def _columns(rows: Sequence[SeriesRow]) -> list[str]:
    """The parameters the template varied, read from the rows.

    Only for a series file that does not record its columns: a run stores the
    template's free parameters, which no repair or filter can change.
    """
    fitted = sorted((row for row in rows if not row.failed), key=lambda row: row.edited)
    return fitted[0].varied() if fitted else []


def _jumps(rows: Sequence[SeriesRow], columns: Sequence[str]) -> dict[int, list[str]]:
    """Reasons per row number for the ``jump`` flag (see ``docs/series.md``).

    Steps are taken between **adjacent** rows that both hold a converged fit,
    so a failed or unconverged row breaks the chain rather than bridging it. A
    step to or from a value held on a bound is left out too: the bound, not the
    data, set that value, and a stretch of such rows would drive the median
    step to zero and make every ordinary step look like a jump.
    """
    reasons: dict[int, list[str]] = {}
    for name in columns:
        steps = []
        for previous, row in itertools.pairwise(rows):
            if not (previous.converged_fit and row.converged_fit):
                continue
            if name in previous.on_bounds or name in row.on_bounds:
                continue
            (before, sigma_before), (after, sigma_after) = (
                previous.estimate(name),
                row.estimate(name),
            )
            if before is None or after is None:
                continue
            sigma = (
                None
                if None in (sigma_before, sigma_after)
                else math.hypot(sigma_before, sigma_after)
            )
            steps.append((row.index, after - before, sigma))
        if len(steps) < JUMP_MIN_STEPS:
            continue
        median = statistics.median(abs(step) for _index, step, _sigma in steps)
        for index, step, sigma in steps:
            if abs(step) > JUMP_MEDIAN_FACTOR * median and (
                sigma is None or abs(step) > JUMP_SIGMA_FACTOR * sigma
            ):
                reasons.setdefault(index, []).append(
                    f'{name} jumped {step:+.3g} (median step {median:.3g})'
                )
    return reasons


def _assess(rows: Sequence[SeriesRow], variable: str, columns: Sequence[str]) -> list[SeriesRow]:
    """The rows with their flags (and display variable) filled in."""
    jumps = _jumps(rows, columns)
    assessed = []
    for row in rows:
        flags = row.own_flags()
        if row.index in jumps:
            flags.append(('jump', '; '.join(jumps[row.index])))
        assessed.append(replace(row, flags=tuple(flags), variable=variable))
    return assessed


# =========================================================================
# Text
# =========================================================================


def _number_text(value: float | None, spec: str = '.4g') -> str:
    return '-' if value is None else format(value, spec)


def _estimate_text(row: SeriesRow, name: str) -> str:
    """``45.12 +/- 0.21``: the uncertainty to two significant figures, the value to match.

    A parameter the row varied without an available uncertainty shows ``± -``;
    one the row held fixed (possible after a repair) shows its value alone.
    """
    value, stderr = row.estimate(name)
    if value is None:
        return '-'
    if row.is_fixed(name):
        return f'{value:.6g}'
    if stderr is None:
        return f'{value:.6g} {PLUS_MINUS} -'
    decimals = max(0, 1 - math.floor(math.log10(stderr)))
    return f'{value:.{decimals}f} {PLUS_MINUS} {stderr:.{decimals}f}'


# =========================================================================
# The result
# =========================================================================


def _kind_mismatch(sesans: bool) -> str:
    """Why a dataset of the other kind cannot join a series of *sesans* data."""
    held, other = ('SESANS', 'I(Q)') if sesans else ('I(Q)', 'SESANS')
    return f'This is {other} data; the series holds {held} data. Fit it as a series of its own.'


class SeriesResult:
    """The outcome of :meth:`IQuana.fit_series`: one row per dataset.

    Renders as a table in a notebook and with ``print()``. Rows are addressed by
    their number in the original run, or by label: ``series[16]``,
    ``series['T_052K.dat']``. See ``docs/series.md`` for the full tour.
    """

    def __init__(
        self,
        rows: Sequence[SeriesRow],
        *,
        variable: str,
        template: dict[str, Any],
        start: str,
        engine: str,
        method: str | None,
        fit_options: dict[str, Any] | None,
        planned: int,
        interrupted: bool = False,
        parameters: Sequence[str] | None = None,
        loaded_from: str | None = None,
        allow_custom_models: bool = False,
    ):
        self.variable = variable
        self.start = start
        self.engine = engine
        self.method = method
        #: The keyword arguments every row's fit() received (as given to fit_series()).
        self.fit_options = dict(fit_options or {})
        self.planned = planned
        self.interrupted = interrupted
        self._template = template
        self._loaded_from = loaded_from
        self._allow_custom_models = allow_custom_models
        #: Set on a filtered selection (flagged()), which is read-only.
        self._selection_of: int | None = None
        #: The table's parameter columns: fixed for the life of the series, so a
        #: repair or a filter never changes the table's shape.
        self._parameters = list(parameters) if parameters is not None else _columns(rows)
        self._set_rows(_assess(rows, variable, self._parameters))

    def _set_rows(self, rows: list[SeriesRow]) -> None:
        self._rows = rows
        self._by_index = {row.index: row for row in rows}
        self._by_label = {row.label: row for row in rows}

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The model expression every row was fitted with."""
        return str(self._template['configuration'].get('model_name'))

    @property
    def parameters(self) -> list[str]:
        """The table's parameter columns: what the template varied."""
        return list(self._parameters)

    @property
    def is_sesans(self) -> bool | None:
        """True for a series of SESANS data, False for I(Q), None when no row has data."""
        for row in self._rows:
            if not row.failed and row.analysis is not None:
                return (row.analysis.get('data') or {}).get('measurement') == 'sesans'
        return None

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[SeriesRow]:
        return iter(self._rows)

    def __getitem__(self, key: int | str) -> SeriesRow:
        """One row, by its number in the original run or by its label."""
        table = self._by_label if isinstance(key, str) else self._by_index
        try:
            return table[key]
        except (KeyError, TypeError):
            raise KeyError(
                f'No row {key!r} in this series. Rows are addressed by their number in '
                'the original run (the # column) or by label.'
            ) from None

    def flagged(self) -> 'SeriesResult':
        """The rows carrying a flag, as a read-only :class:`SeriesResult`.

        Row numbers stay the original ones, so ``series.flagged()[16]`` is row
        16. Printing, plots, ``to_dataframe()`` and ``save_table()`` work on it;
        ``update()`` and ``save_analysis()`` belong to the full series.
        """
        selection = copy.copy(self)
        selection._selection_of = (
            len(self._rows) if self._selection_of is None else self._selection_of
        )
        selection._set_rows([row for row in self._rows if row.flags])
        return selection

    def warnings(self) -> dict[str, int]:
        """Each distinct warning text raised while fitting, with the number of rows."""
        counts = Counter(text for row in self._rows for text in set(row.warnings))
        return dict(counts.most_common())

    def _counts(self) -> dict[str, int]:
        """``{'ok', 'flagged', 'failed'}``: rows by outcome; each row counts once."""
        failed = sum(row.failed for row in self._rows)
        flagged = sum(bool(row.flags) and not row.failed for row in self._rows)
        return {'ok': len(self._rows) - failed - flagged, 'flagged': flagged, 'failed': failed}

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _fields(self) -> list[str]:
        """The keys of every :meth:`table` row, in order; the same for an empty table."""
        pairs = (key for name in self._parameters for key in (name, f'{name}_err'))
        return ['row', 'label', self.variable, *pairs, *TABLE_COLUMNS[2:]]

    def table(self) -> list[dict[str, Any]]:
        """One flat dict per row, the representation every export is built from.

        Keys: ``row``, ``label``, the variable, each parameter and its
        ``<name>_err``, ``chisq``, ``reduced_chisq``, ``n_points``, ``flags``
        (comma-separated), ``edited`` and ``error``. A missing number is None.
        """
        columns = self.parameters
        table = []
        for row in self._rows:
            entry: dict[str, Any] = {'row': row.index, 'label': row.label, self.variable: row.value}
            for name in columns:
                value, stderr = row.estimate(name)
                entry[name] = value
                entry[f'{name}_err'] = None if row.is_fixed(name) else stderr
            entry.update(
                chisq=row.chisq,
                reduced_chisq=row.reduced_chisq,
                n_points=row.n_points,
                flags=', '.join(row.flag_names),
                edited=row.edited,
                error=row.error,
            )
            table.append(entry)
        return table

    def to_dataframe(self) -> Any:
        """:meth:`table` as a pandas DataFrame indexed by row number.

        pandas is an optional dependency and is imported here only.
        """
        import pandas as pd

        return pd.DataFrame(self.table(), columns=self._fields()).set_index('row')

    def save_table(self, filename: str) -> None:
        """Write :meth:`table` as CSV, including each failure's reason.

        Not ``save_results``: ``IQuana.save_results()`` writes one fit's curve
        and residuals, which is a different kind of file.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self._fields())
        writer.writeheader()
        writer.writerows(self.table())
        atomic_write(filename, buffer.getvalue())
        logger.info(f'{OK} Series table saved to {filename}')

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _header(self) -> str:
        what = (
            f'{len(self._rows)} datasets'
            if self._selection_of is None
            else f'{len(self._rows)} of {self._selection_of} rows flagged'
        )
        return (
            f'Series fit {MIDDLE_DOT} {self.model} {MIDDLE_DOT} {what} {MIDDLE_DOT} '
            f'variable: {self.variable} {MIDDLE_DOT} start={self.start}'
        )

    def _summary(self) -> str:
        counts = self._counts()
        text = f'{counts["ok"]} ok, {counts["flagged"]} flagged, {counts["failed"]} failed'
        if self.interrupted:
            text += f' {MIDDLE_DOT} {self._processed()} of {self.planned} processed, interrupted'
        return text

    def _processed(self) -> int:
        return len(self._rows) if self._selection_of is None else self._selection_of

    def _grid(self, symbol: str) -> tuple[list[str], list[list[str]]]:
        """Header and cells shared by the text and HTML tables.

        A selection (``flagged()``) shows the reason instead of the parameters.
        """
        if self._selection_of is not None:
            header = ['#', 'label', self.variable, symbol, 'why']
            cells = [
                [
                    str(row.index),
                    row.label,
                    f'{row.value:g}',
                    _number_text(row.reduced_chisq),
                    row.why(),
                ]
                for row in self._rows
            ]
            return header, cells
        columns = self.parameters
        header = ['#', 'label', self.variable, *columns, symbol, 'flags']
        cells = []
        for row in self._rows:
            marks = list(row.flag_names) + (['edited'] if row.edited else [])
            cells.append(
                [str(row.index), row.label, f'{row.value:g}']
                + [_estimate_text(row, name) for name in columns]
                + [_number_text(row.reduced_chisq), ', '.join(marks)]
            )
        return header, cells

    def _abbreviated(
        self, cells: list[list[str]]
    ) -> tuple[list[list[str]] | None, list[list[str]] | None]:
        """The first and last rows when there are too many; ``(cells, None)`` otherwise."""
        if len(cells) <= DISPLAY_ROWS:
            return cells, None
        half = DISPLAY_ROWS // 2
        return cells[:half], cells[-half:]

    def __str__(self) -> str:
        header, cells = self._grid(REDUCED_CHI_SQUARED)
        widths = [max(len(text) for text in column) for column in zip(header, *cells, strict=True)]

        def line(values: list[str]) -> str:
            parts = [
                value.ljust(width) if position in (1, len(values) - 1) else value.rjust(width)
                for position, (value, width) in enumerate(zip(values, widths, strict=True))
            ]
            return '  '.join(parts).rstrip()

        head, tail = self._abbreviated(cells)
        out = [self._header(), self._summary(), '', line(header)]
        out += [line(values) for values in head]
        if tail is not None:
            out.append(' ...')
            out += [line(values) for values in tail]
            out += [
                '',
                f'Showing {DISPLAY_ROWS} of {len(cells)} rows; table() or to_dataframe() has all of them.',
            ]
        return '\n'.join(out)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        esc = escape_html_cell
        header, cells = self._grid('χ²ᵣ')
        cell = 'padding:2px 8px;border:1px solid #ddd;text-align:right'
        head = 'padding:2px 8px;border:1px solid #ddd;text-align:left;background:#f5f5f5'
        flagged_rows = {str(row.index) for row in self._rows if row.flags}

        def html_row(values: list[str]) -> str:
            tint = ';background:#fff4e5' if values[0] in flagged_rows else ''
            return (
                '<tr>'
                + ''.join(f'<td style="{cell}{tint}">{esc(v)}</td>' for v in values)
                + '</tr>'
            )

        head_cells, tail_cells = self._abbreviated(cells)
        parts = [
            f'<div><p><strong>{esc(self._header())}</strong><br>{esc(self._summary())}</p>',
            '<table style="border-collapse:collapse">',
            '<tr>' + ''.join(f'<th style="{head}">{esc(h)}</th>' for h in header) + '</tr>',
            *(html_row(values) for values in head_cells),
        ]
        if tail_cells is not None:
            parts.append(
                f'<tr><td colspan="{len(header)}" style="{cell};text-align:center">...</td></tr>'
            )
            parts += [html_row(values) for values in tail_cells]
        parts.append('</table>')
        if tail_cells is not None:
            parts.append(
                f'<p>Showing {DISPLAY_ROWS} of {len(cells)} rows; table() or to_dataframe() has all of them.</p>'
            )
        parts.append('</div>')
        return ''.join(parts)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot(self, *parameters: str, show: bool | None = None) -> Any:
        """Trend plot: each parameter, and reduced χ² at the bottom, against the variable.

        Args:
            *parameters: The panels to draw; all varied parameters when omitted.
            show: As for every IQuana plot: ``None`` shows the figure except in
                a notebook, where the returned figure renders itself.

        Returns:
            The plotly figure.
        """
        unknown = [name for name in parameters if name not in self.parameters]
        if unknown:
            raise ValueError(
                f'Unknown parameter(s) {", ".join(unknown)}. The series varied: '
                f'{", ".join(self.parameters) or "nothing (no row has a fit)"}.'
            )
        return series_plotting.plot_trend(self, list(parameters) or self.parameters, show=show)

    def plot_fits(
        self, rows: Sequence[int | str] | int | str | None = None, *, show: bool | None = None
    ) -> Any:
        """Data and fit for several rows, overlaid.

        Args:
            rows: A row number or label, a list of them, or ``'flagged'``. By
                default up to nine rows spread evenly over the rows that have a
                fit.
            show: As for :meth:`plot`.

        A row whose saved result cannot be restored (its data file changed) is
        drawn as the model at its stored parameters, labelled as a theory curve.
        A row that cannot be drawn at all is named with the reason.
        """
        if rows is None:
            fitted = [row.index for row in self._rows if not row.failed]
            count = min(DEFAULT_FIT_PLOTS, len(fitted))
            chosen = [
                fitted[round(i * (len(fitted) - 1) / max(count - 1, 1))] for i in range(count)
            ]
            selected = [self[index] for index in dict.fromkeys(chosen)]
            if not selected:
                raise ValueError('No rows to plot: no row of this series has a fit.')
        elif rows == 'flagged':
            selected = list(self.flagged())
            if not selected:
                raise ValueError('No rows to plot: no row of this series is flagged.')
        else:
            keys = [rows] if isinstance(rows, (int, str)) else rows
            selected = [self[key] for key in keys]
        return series_plotting.plot_fits(self, selected, show=show)

    # ------------------------------------------------------------------
    # Drill-down and repair
    # ------------------------------------------------------------------

    def fitter(self, row: int | str) -> IQuana:
        """A normal ``IQuana`` holding one row's data, setup and result.

        A fitted row comes back as it was after its fit. A row that loaded but
        failed comes back with its data and the setup of the attempt. A row
        whose file never loaded gets the template setup with the file loaded
        now, so this raises the load error again while the file is broken, and
        is the retry path once it is fixed.
        """
        target = self[row]
        if target.analysis is not None:
            return fitter_from_document(
                IQuana,
                target.analysis,
                self._loaded_from,
                allow_custom_models=self._allow_custom_models,
            )
        if target.source is None:
            raise ValueError(
                f'Row {target.index} is an in-memory dataset that could not be set '
                f'({target.error}); there is nothing to reopen.'
            )
        return fitter_from_document(
            IQuana,
            self._template,
            data=self._source_path(target),
            allow_custom_models=self._allow_custom_models,
        )

    def _source_path(self, row: SeriesRow) -> str | None:
        """Where *row*'s file is now: next to the series file first, then where it was."""
        if row.source is None:
            return None
        section = {'path_absolute': row.source, 'path_relative': row.source_relative}
        found, _tried = locate_data_file(section, self._loaded_from)
        return found or row.source

    def update(self, row: int | str, fitter: IQuana) -> None:
        """Replace a row with a repaired fit, then recompute the flags.

        Checked before anything is replaced: the fitter holds a result that
        describes its current setup, its data is the row's data (1D, as for the
        run), and its model is the series' model. Values, bounds and vary flags
        may differ from the template. The row is marked edited.

        A repair fitted with another engine or method than the rest of the
        series is accepted with a warning: its χ² and uncertainties are not
        comparable with theirs. ``fitter.fit(series.engine, series.method,
        **series.fit_options)`` repeats the series' own fit.

        Raises:
            ValueError: If any check fails; the row is left unchanged.
        """
        self._require_full('update()')
        target = self[row]
        if fitter.is_2d:
            raise ValueError('2D data is not supported by fit_series() in this version.')
        document = build_analysis_dict(fitter, None)
        if document['result'] is None:
            reason = document['result_omitted'] or 'the fitter holds no fit result'
            raise ValueError(
                f'Fit this dataset again before updating row {target.index} ({reason}).'
            )
        self._check_same_data(target, document)
        self._check_same_kind(document)
        self._check_same_model(target, document)
        self._warn_if_other_optimizer(target, document['result'])
        repaired = replace(target, analysis=document, error=None, warnings=(), edited=True)
        rows = [repaired if r.index == target.index else r for r in self._rows]
        self._set_rows(_assess(rows, self.variable, self._parameters))
        logger.info(f'{OK} Row {target.index} ({target.label}) replaced; flags recomputed')

    def _warn_if_other_optimizer(self, target: SeriesRow, result: dict[str, Any]) -> None:
        reference = next((r.result for r in self._rows if not r.failed and not r.edited), None)
        if reference is None:
            return
        used = (result.get('engine'), result.get('method'))
        expected = (reference.get('engine'), reference.get('method'))
        if used != expected:
            warnings.warn(
                f'Row {target.index} was refitted with {"/".join(map(str, used))}; the series '
                f'was fitted with {"/".join(map(str, expected))}. Their chi-squared values and '
                'uncertainties are not comparable. fitter.fit(series.engine, series.method, '
                '**series.fit_options) repeats the series fit.',
                stacklevel=3,
            )

    def _check_same_data(self, target: SeriesRow, document: dict[str, Any]) -> None:
        data = document['data'] or {}
        if target.analysis is not None:
            expected = (target.analysis.get('data') or {}).get('array_fingerprint_now')
            same = data.get('array_fingerprint_now') == expected
        else:
            same = (
                data.get('source') == 'file'
                and os.path.normcase(os.path.abspath(data.get('path_absolute') or ''))
                == os.path.normcase(self._source_path(target) or '')
                and data.get('dataset') in (0, None)
            )
        if not same:
            raise ValueError(
                f'The fitter does not hold the data of row {target.index} ({target.label}). '
                f'Open the row with series.fitter({target.index}), repair it there, and update with that fitter.'
            )

    def _check_same_kind(self, document: dict[str, Any]) -> None:
        sesans = self.is_sesans
        if sesans is not None and ((document['data'] or {}).get('measurement') == 'sesans') != (
            sesans
        ):
            raise ValueError(_kind_mismatch(sesans))

    def _check_same_model(self, target: SeriesRow, document: dict[str, Any]) -> None:
        def model(config: dict[str, Any]) -> tuple[Any, ...]:
            factor = config.get('structure_factor') or {}
            return config.get('model_name'), config.get('components'), factor.get('name')

        if model(document['configuration']) != model(self._template['configuration']):
            raise ValueError(
                f'Row {target.index} was fitted with {self.model}; the fitter uses a different '
                'model. A series has one model: fit the other model as a second series.'
            )

    def _require_full(self, action: str) -> None:
        if self._selection_of is not None:
            raise ValueError(
                f'This is a read-only selection (flagged()). Call {action} on the full series.'
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_analysis(self, filename: str) -> None:
        """Save the whole series as JSON: each row a complete single analysis.

        Data files are referenced, not copied; in-memory datasets are embedded.
        Each data path is written relative to *filename* too, so a series moved
        together with its data still finds it.
        """
        self._require_full('save_analysis()')
        target = os.path.abspath(filename)
        directory = os.path.dirname(target)
        if directory and not os.path.isdir(directory):
            raise AnalysisFileError(f'Directory does not exist: {directory}')
        document = {
            'schema': {
                'format': SERIES_FORMAT,
                'version': SERIES_VERSION,
                'written_by': __version__,
            },
            'variable': self.variable,
            'start': self.start,
            'engine': self.engine,
            'method': self.method,
            'fit_options': self._recordable_options(),
            'planned': self.planned,
            'processed': len(self._rows),
            'interrupted': self.interrupted,
            'parameters': self._parameters,
            'template': self._template,
            'rows': [self._row_document(row, target) for row in self._rows],
        }
        atomic_write(target, json.dumps(document, indent=1, allow_nan=False, ensure_ascii=False))
        logger.info(f'{OK} Series saved to {filename} ({len(self._rows)} rows)')

    def _recordable_options(self) -> dict[str, Any] | None:
        """The fit options as the file stores them, or None when JSON cannot hold them."""
        try:
            json.dumps(self.fit_options, allow_nan=False)
        except (TypeError, ValueError):
            warnings.warn(
                'The fit options of this series are not plain JSON values, so the saved '
                'file does not record them.',
                stacklevel=3,
            )
            return None
        return self.fit_options

    def _row_document(self, row: SeriesRow, target: str) -> dict[str, Any]:
        analysis = row.analysis
        if analysis is not None:
            data = relocate_data_section(analysis.get('data'), self._loaded_from, target)
            analysis = {**analysis, 'data': data}
        source = row.source
        if source is not None:  # the retry path of a row whose file never loaded
            source = relocate_data_section(
                {'source': 'file', 'path_absolute': source, 'path_relative': row.source_relative},
                self._loaded_from,
                target,
            )
        return {
            'label': row.label,
            'value': row.value,
            'source': None if source is None else source['path_absolute'],
            'source_relative': None if source is None else source['path_relative'],
            'start_row': row.start_row,
            'edited': row.edited,
            'error': row.error,
            'warnings': list(row.warnings),
            'analysis': analysis,
        }

    @classmethod
    def load_analysis(cls, filename: str, allow_custom_models: bool = False) -> 'SeriesResult':
        """Reopen a series written by :meth:`save_analysis`.

        Reads the JSON only: the table, flags and trend plot need no data
        files. A row's data file is found and checked when the row is opened
        with :meth:`fitter` or drawn with :meth:`plot_fits`.

        Args:
            filename: Path to the series file.
            allow_custom_models: Permit a model expression that is not built
                into sasmodels, applied whenever a row is opened. Off by
                default, as for ``IQuana.load_analysis()``.

        Raises:
            AnalysisFileError: If the file is missing or malformed, is a single
                analysis, or holds a document from a newer IQuana.
        """
        document = read_json(filename, 'Series file')
        check_schema(document, SERIES_FORMAT, SERIES_VERSION, (SERIES_VERSION,))
        template = document.get('template')
        check_analysis_document(template, 'template.')
        entries = document.get('rows')
        _require(isinstance(entries, list), 'rows', 'must be a list')
        variable = document.get('variable')
        _require(
            isinstance(variable, str) and variable.isidentifier(), 'variable', 'must be a name'
        )
        _require(document.get('start') in ('previous', 'template'), 'start', 'is not a start mode')
        _require(isinstance(document.get('engine'), str), 'engine', 'must be a string')
        planned = document.get('planned')
        _require(
            isinstance(planned, int) and planned >= len(entries),
            'planned',
            'must be a count no smaller than the number of rows',
        )
        parameters = document.get('parameters')
        _require(
            parameters is None
            or (isinstance(parameters, list) and all(isinstance(p, str) for p in parameters)),
            'parameters',
            'must be a list of names',
        )
        rows = [_row_from_document(index, entry) for index, entry in enumerate(entries)]
        labels = Counter(row.label for row in rows)
        _require(
            all(n == 1 for n in labels.values()),
            'rows',
            f'repeat a label ({", ".join(sorted(k for k, n in labels.items() if n > 1))})',
        )
        series = cls(
            rows,
            variable=variable,
            template=template,
            start=document['start'],
            engine=document['engine'],
            method=document.get('method'),
            fit_options=document.get('fit_options'),
            planned=planned,
            interrupted=bool(document.get('interrupted')),
            parameters=parameters,
            loaded_from=os.path.abspath(filename),
            allow_custom_models=allow_custom_models,
        )
        note = (
            f', {len(entries)} of {series.planned} processed, interrupted'
            if series.interrupted
            else ''
        )
        logger.info(f'{OK} Series loaded from {filename}: {len(entries)} rows{note}')
        return series


def _row_from_document(index: int, entry: Any) -> SeriesRow:
    """One stored row, its envelope held to the rules of a single analysis file."""
    where = f'rows[{index}]'
    _require(isinstance(entry, dict), where, 'must be a mapping')
    _require(isinstance(entry.get('label'), str), f'{where}.label', 'must be a string')
    _require(_finite(entry.get('value')) is not None, f'{where}.value', 'must be a finite number')
    analysis = entry.get('analysis')
    if analysis is not None:
        check_analysis_document(analysis, f'{where}.analysis.')
    start_row = entry.get('start_row')
    _require(
        start_row is None or (isinstance(start_row, int) and 0 <= start_row < index),
        f'{where}.start_row',
        'must be the number of an earlier row, or null',
    )
    for key in ('source', 'source_relative', 'error'):
        _require(
            entry.get(key) is None or isinstance(entry.get(key), str),
            f'{where}.{key}',
            'must be a string or null',
        )
    return SeriesRow(
        index=index,
        label=entry['label'],
        value=float(entry['value']),
        source=entry.get('source'),
        analysis=analysis,
        error=entry.get('error'),
        warnings=tuple(str(text) for text in entry.get('warnings') or ()),
        start_row=start_row,
        edited=bool(entry.get('edited')),
        source_relative=entry.get('source_relative'),
    )
