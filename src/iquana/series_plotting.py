"""Figures for a series fit: parameter trends, and data with fits for chosen rows.

Both take a :class:`~iquana.series_results.SeriesResult` and are reached
through its ``plot()`` and ``plot_fits()`` methods. They read the rows only;
this module does not import :mod:`iquana.series_results`, which imports it.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .console import raised_level
from .multi_plotting import DatasetPanel, plot_multi
from .plotting import resolve_show

__all__ = ['plot_trend', 'plot_fits']

#: Vertical pixels per trend panel.
TREND_PANEL_HEIGHT_PIXELS = 220
#: The panel name of the goodness of fit; it is always drawn, at the bottom.
CHI_PANEL = 'reduced_chisq'
OK_COLOR = '#636efa'
FLAGGED_COLOR = '#ef553b'
FAILED_COLOR = 'gray'


def plot_trend(series: Any, parameters: Sequence[str], *, show: bool | None = None) -> go.Figure:
    """Stacked panels of each parameter, then reduced χ², over the series variable.

    Points are joined in series order, so a loop is drawn as a loop. The line
    breaks at a failed or unconverged row. Flagged rows use a different marker
    and give their reason on hover; failed rows, which have no value, are
    marked with × on the x axis of the bottom panel.
    """
    rows = list(series)
    panels = [*parameters, CHI_PANEL]
    figure = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=min(0.04, 0.3 / len(panels))
    )
    for position, name in enumerate(panels, start=1):
        for trace in _trend_traces(rows, name):
            figure.add_trace(trace, row=position, col=1)
        figure.update_yaxes(title_text='χ²ᵣ' if name == CHI_PANEL else name, row=position, col=1)

    failed = [row for row in rows if row.failed]
    if failed:
        chi = [row.reduced_chisq for row in rows if row.reduced_chisq is not None]
        figure.add_trace(
            go.Scatter(
                x=[row.value for row in failed],
                y=[min(chi, default=0.0)] * len(failed),
                mode='markers',
                marker={'symbol': 'x', 'size': 10, 'color': FAILED_COLOR},
                hovertext=[f'#{row.index} {row.label}: {row.failure}' for row in failed],
                hoverinfo='text',
                name='failed',
            ),
            row=len(panels),
            col=1,
        )
    figure.update_xaxes(title_text=series.variable, row=len(panels), col=1)
    title = f'{series.model}: {len(rows)} datasets'
    if failed:
        title += f' ({len(failed)} failed, marked ×)'
    figure.update_layout(
        title=title,
        template='plotly_white',
        showlegend=False,
        height=TREND_PANEL_HEIGHT_PIXELS * len(panels) + 120,
    )
    if resolve_show(show):
        figure.show()
    return figure


def _trend_traces(rows: Sequence[Any], name: str) -> list[go.Scatter]:
    def estimate(row: Any) -> tuple[float | None, float | None]:
        return (row.reduced_chisq, None) if name == CHI_PANEL else row.estimate(name)

    estimates = [estimate(row) for row in rows]
    line = go.Scatter(
        x=[row.value for row in rows],
        y=[
            value if row.converged_fit else None
            for row, (value, _err) in zip(rows, estimates, strict=True)
        ],
        mode='lines',
        line={'color': 'lightgray', 'width': 1},
        connectgaps=False,
        hoverinfo='skip',
    )
    traces = [line]
    for flagged, color, symbol in ((False, OK_COLOR, 'circle'), (True, FLAGGED_COLOR, 'diamond')):
        chosen = [
            (row, value, err)
            for row, (value, err) in zip(rows, estimates, strict=True)
            if value is not None and not row.failed and bool(row.flags) == flagged
        ]
        if not chosen:
            continue
        traces.append(
            go.Scatter(
                x=[row.value for row, _v, _e in chosen],
                y=[value for _r, value, _e in chosen],
                error_y={
                    'type': 'data',
                    'array': [np.nan if err is None else err for _r, _v, err in chosen],
                    'visible': True,
                },
                mode='markers',
                marker={'color': color, 'symbol': symbol, 'size': 8 if flagged else 6},
                hovertext=[
                    f'#{row.index} {row.label}' + (f'<br>{row.why()}' if flagged else '')
                    for row, _v, _e in chosen
                ],
                hoverinfo='text+x+y',
                name='flagged' if flagged else name,
            )
        )
    return traces


def plot_fits(series: Any, rows: Sequence[Any], *, show: bool | None = None) -> go.Figure:
    """Data and fit for *rows*, overlaid, through :func:`multi_plotting.plot_multi`.

    A row that cannot be drawn is named with its reason: in a warning when
    others can be drawn, in the error when none can.
    """
    panels, problems = [], []
    for row in rows:
        try:
            panels.append(_fit_panel(series, row))
        except Exception as exc:  # noqa: BLE001 - one row must not stop the figure
            problems.append(f'#{row.index} {row.label}: {exc}')
    if not panels:
        raise ValueError('None of the rows can be drawn:\n  ' + '\n  '.join(problems))
    if problems:
        warnings.warn('Not drawn: ' + '; '.join(problems), stacklevel=3)
    return plot_multi(
        panels,
        title=f'{series.model}: {len(panels)} of {len(series)} datasets',
        overlay=True,
        show=show,
    )


def _fit_panel(series: Any, row: Any) -> DatasetPanel:
    """One row's data and curve. The fitted curve when the saved result restores;
    otherwise the model at the stored values, labelled as a theory curve."""
    if row.failed:
        raise ValueError(f'failed: {row.failure}')
    with raised_level():  # opening a row logs its loading; nine of them would be noise
        fitter = series.fitter(row.index)
    name = f'#{row.index} {row.label} · {row.value:g}'
    artifacts = fitter._fit_contract.artifacts if fitter._fit_contract is not None else None
    if artifacts is not None and artifacts.fitted_curve is not None:
        curve, index, residuals = artifacts.fitted_curve, artifacts.fit_index, artifacts.residuals
    else:
        curve, index = fitter._evaluate(fitter._evaluation_data(warn=False))
        residuals = None
        name += ' (theory at saved values)'
    return DatasetPanel.from_dataset(
        name=name,
        model=series.model,
        data=fitter.data,
        fitted_curve=curve,
        fit_index=index,
        residuals=residuals,
    )
