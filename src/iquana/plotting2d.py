"""Heatmap displays for 2D data: data, model and residual maps.

The fit works on flat per-pixel vectors; only the display grids them. A file
that is a regular Cartesian grid is reshaped exactly. Anything else (the
S50 magnetic files, for instance, whose ``x_bins`` has a single entry) is
binned onto square bins as a mean, with empty bins left blank. Binning never
touches the fit's residual vector or its degrees of freedom.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data.view import intensity, uncertainty
from .results import FitResultContract, resolve_fit_index

PREVIEW_TITLE_PREFIX = 'Model preview'
RESIDUAL_COLORSCALE = 'RdBu_r'
INTENSITY_COLORSCALE = 'Viridis'


@dataclass(slots=True)
class Grid:
    """Image axes, and the image cell (row, column) each pixel falls in."""

    qx: np.ndarray
    qy: np.ndarray
    shape: tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray

    def image(self, values: np.ndarray) -> np.ndarray:
        """Mean of the finite *values* in each cell; NaN where a cell has none."""
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        total = np.zeros(self.shape)
        count = np.zeros(self.shape)
        np.add.at(total, (self.rows[finite], self.cols[finite]), values[finite])
        np.add.at(count, (self.rows[finite], self.cols[finite]), 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(count > 0, total / count, np.nan)


def _bins(values: Any) -> np.ndarray:
    return np.zeros(0) if values is None else np.asarray(values, dtype=float).ravel()


def build_grid(data: Any) -> Grid:
    """The display grid for *data*: an exact reshape when possible, else bins."""
    qx = np.asarray(data.qx_data, dtype=float)
    qy = np.asarray(data.qy_data, dtype=float)
    n = qx.size
    x_bins = _bins(getattr(data, 'x_bins', None))
    y_bins = _bins(getattr(data, 'y_bins', None))
    nx, ny = x_bins.size, y_bins.size
    pixel = np.arange(n)
    if nx > 1 and ny > 1 and nx * ny == n:
        # A regular grid only if the coordinates really are the mesh, in one
        # of the two pixel orders. A length check alone accepts a 1-wide
        # "grid" of irregular points.
        if np.allclose(qx, np.tile(x_bins, ny)) and np.allclose(qy, np.repeat(y_bins, nx)):
            return Grid(x_bins, y_bins, (ny, nx), pixel // nx, pixel % nx)
        if np.allclose(qx, np.repeat(x_bins, ny)) and np.allclose(qy, np.tile(y_bins, nx)):
            return Grid(x_bins, y_bins, (ny, nx), pixel % ny, pixel // ny)

    # Square bins, sized as sasmodels' own plotter sizes them.
    ny = max(int(np.sqrt(n)), 1)
    nx = max(n // ny, 1)
    x_edges = np.linspace(qx.min(), qx.max(), nx + 1)
    y_edges = np.linspace(qy.min(), qy.max(), ny + 1)
    cols = np.clip(np.searchsorted(x_edges, qx, side='right') - 1, 0, nx - 1)
    rows = np.clip(np.searchsorted(y_edges, qy, side='right') - 1, 0, ny - 1)
    centres = (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2
    return Grid(centres[0], centres[1], (ny, nx), rows, cols)


def _log_limits(values: np.ndarray, sigma: np.ndarray | None) -> tuple[float, float]:
    """Colour range in log10, from the data as sasmodels' plotter computes it."""
    upper = np.abs(values) + (0 if sigma is None else np.nan_to_num(sigma))
    upper = upper[np.isfinite(upper) & (upper > 0)]
    if upper.size == 0:
        return -1.0, 1.0
    low, high = np.log10(upper.min() / 4), np.log10(upper.max())
    return (low, high) if high > low else (low - 1, high + 1)


def _intensity_panel(image: np.ndarray, log_scale: bool, limits: tuple[float, float]) -> dict:
    """z values and colour-axis settings for a data or model map.

    On a log axis non-positive intensities are clipped to the bottom of the
    range, so they stay visible: only excluded pixels are transparent.
    """
    if not log_scale:
        return {'z': image, 'customdata': image}
    floor = 10 ** limits[0]
    with np.errstate(invalid='ignore'):
        clipped = np.where(np.isnan(image), np.nan, np.maximum(image, floor))
        return {'z': np.log10(clipped), 'customdata': image}


def plot_fit_2d(
    data: Any,
    fit_result: FitResultContract | None,
    model_name: str | None,
    *,
    included: np.ndarray,
    show_residuals: bool = True,
    log_scale: bool = True,
    show_components: bool = False,
    weighting_mode: str = 'dI',
) -> go.Figure:
    """Data, model and residual maps for 2D data.

    Args:
        data: The fitter's dataset (its own intensity and dI).
        fit_result: The fit or preview to draw; None draws the data alone.
        included: Pixels the fit selects (or would select), as a boolean
            array; every other pixel is drawn transparent.
        log_scale: Log colour axis for intensities. The q axes are always
            linear and equally scaled.
    """
    if show_components:
        warnings.warn(
            'Component maps are not available for 2D data; show_components is ignored.',
            stacklevel=3,
        )
    grid = build_grid(data)
    values = intensity(data)
    sigma = uncertainty(data)
    observed = np.where(included, values, np.nan)
    limits = _log_limits(observed, None if sigma is None else np.where(included, sigma, np.nan))

    panels: list[tuple[str, dict, str]] = [
        ('Data', _intensity_panel(grid.image(observed), log_scale, limits), 'coloraxis')
    ]
    title = 'SANS Data (2D)'
    if fit_result is not None:
        index = resolve_fit_index(fit_result.artifacts.fit_index, values.size)
        model = np.full(values.size, np.nan)
        model[index] = fit_result.require_fitted_curve()
        panels.append(
            ('Model', _intensity_panel(grid.image(model), log_scale, limits), 'coloraxis')
        )
        residuals = fit_result.artifacts.residuals
        if show_residuals and residuals is not None and len(residuals) == int(index.sum()):
            full = np.full(values.size, np.nan)
            full[index] = residuals
            image = grid.image(full)
            panels.append(('Residuals', {'z': image, 'customdata': image}, 'coloraxis2'))
        from .plotting import format_reduced_chisq

        prefix = PREVIEW_TITLE_PREFIX if fit_result.is_preview else 'SANS Fit'
        title = f'{prefix}: {model_name} ({format_reduced_chisq(fit_result.reduced_chisq, fit_result.dof)})'

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[name for name, _panel, _axis in panels],
        horizontal_spacing=0.08,
    )
    unit = 'intensity units' if weighting_mode == 'none' else 'σ'
    for column, (name, panel, axis) in enumerate(panels, start=1):
        label = 'residual' if name == 'Residuals' else 'I'
        fig.add_trace(
            go.Heatmap(
                x=grid.qx,
                y=grid.qy,
                z=panel['z'],
                customdata=panel['customdata'],
                coloraxis=axis,
                hovertemplate=f'qx=%{{x:.4g}}<br>qy=%{{y:.4g}}<br>{label}=%{{customdata:.4g}}<extra>{name}</extra>',
            ),
            row=1,
            col=column,
        )
        anchor = 'x' if column == 1 else f'x{column}'
        fig.update_xaxes(title_text='Qx (Å⁻¹)', row=1, col=column)
        fig.update_yaxes(scaleanchor=anchor, scaleratio=1, row=1, col=column)
    fig.update_yaxes(title_text='Qy (Å⁻¹)', row=1, col=1)

    layout = {
        'coloraxis': {
            'colorscale': INTENSITY_COLORSCALE,
            'cmin': limits[0] if log_scale else None,
            'cmax': limits[1] if log_scale else None,
            'colorbar': {
                'title': 'log₁₀ I' if log_scale else 'I',
                'x': 1.0 if len(panels) < 3 else 0.64,
            },
        },
    }
    if any(axis == 'coloraxis2' for _name, _panel, axis in panels):
        span = (
            float(np.nanmax(np.abs(panels[-1][1]['z'])))
            if np.isfinite(panels[-1][1]['z']).any()
            else 1.0
        )
        layout['coloraxis2'] = {
            'colorscale': RESIDUAL_COLORSCALE,
            'cmin': -span,
            'cmax': span,
            'colorbar': {'title': f'residual ({unit})', 'x': 1.02},
        }
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=450,
        width=420 * len(panels) + 120,
        **layout,
    )
    return fig


# =============================================================================
# 2D to 1D: the regions of the cuts, and the curves
# =============================================================================


def plot_cuts(
    data: Any,
    *cuts: Any,
    excluded: np.ndarray,
    log_scale: bool = True,
    show: bool | None = None,
) -> go.Figure:
    """The image with each cut's region outlined, next to the cut curves.

    Args:
        data: The 2D dataset the cuts were made from.
        cuts: Results of ``circular_average``, ``sector_average`` (any
            ``Data1D``) and ``annular_average``.
        excluded: Pixels drawn transparent (True = excluded).
        log_scale: Log colour axis for the image, and log-log I(Q).
    """
    from plotly.colors import qualitative

    from .data.averaging import AnnularAverage
    from .data.selection import coordinate_digest
    from .plotting import error_bars, resolve_show

    if not cuts:
        raise ValueError('Pass at least one cut to plot_cuts().')
    rings = [cut for cut in cuts if isinstance(cut, AnnularAverage)]
    curves = [cut for cut in cuts if not isinstance(cut, AnnularAverage)]
    titles = ['Regions'] + (['I(Q)'] if curves else []) + (['I(φ)'] if rings else [])
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, horizontal_spacing=0.1)

    grid = build_grid(data)
    observed = np.where(excluded, np.nan, intensity(data))
    sigma = uncertainty(data)
    limits = _log_limits(observed, None if sigma is None else np.where(excluded, np.nan, sigma))
    panel = _intensity_panel(grid.image(observed), log_scale, limits)
    fig.add_trace(
        go.Heatmap(
            x=grid.qx,
            y=grid.qy,
            z=panel['z'],
            customdata=panel['customdata'],
            coloraxis='coloraxis',
            hovertemplate='qx=%{x:.4g}<br>qy=%{y:.4g}<br>I=%{customdata:.4g}<extra></extra>',
        ),
        row=1,
        col=1,
    )

    digest = coordinate_digest(data)
    colours = qualitative.Plotly
    curve_column = 2
    ring_column = 2 + bool(curves)
    for number, cut in enumerate(cuts):
        colour = colours[number % len(colours)]
        is_ring = isinstance(cut, AnnularAverage)
        record = None if is_ring else getattr(cut, 'averaging', None)
        name = cut.title if is_ring else str(getattr(cut, 'title', '') or f'cut {number + 1}')
        group = {
            'legendgroup': f'cut{number}',
            'line': {'color': colour},
            'marker': {'color': colour},
        }
        source = cut.image if is_ring else (record or {}).get('image')
        if not is_ring and not isinstance(record, dict):
            warnings.warn(
                f"'{name}' has no averaging record (it was rebuilt, for example by "
                'load_analysis() or arithmetic), so its region is not drawn.',
                stacklevel=3,
            )
        elif source != digest:
            warnings.warn(
                f"'{name}' was not made from this image: its region is drawn on the wrong pixels.",
                stacklevel=3,
            )
        for x, y in _outlines(cut, record):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', showlegend=False, hoverinfo='skip', **group),
                row=1,
                col=1,
            )
        if is_ring:
            fig.add_trace(
                go.Scatter(
                    x=cut.phi,
                    y=cut.intensity,
                    error_y=error_bars(cut.uncertainty),
                    mode='markers+lines',
                    name=name,
                    **group,
                ),
                row=1,
                col=ring_column,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=cut.x,
                    y=cut.y,
                    error_y=error_bars(cut.dy),
                    mode='markers',
                    name=name,
                    **group,
                ),
                row=1,
                col=curve_column,
            )

    fig.update_xaxes(title_text='Qx (Å⁻¹)', row=1, col=1)
    fig.update_yaxes(title_text='Qy (Å⁻¹)', scaleanchor='x', scaleratio=1, row=1, col=1)
    if curves:
        axis_type = 'log' if log_scale else 'linear'
        fig.update_xaxes(title_text='Q (Å⁻¹)', type=axis_type, row=1, col=curve_column)
        fig.update_yaxes(title_text='I(Q)', type=axis_type, row=1, col=curve_column)
    if rings:
        fig.update_xaxes(title_text='φ (degrees)', dtick=90, row=1, col=ring_column)
        fig.update_yaxes(title_text='I(φ)', row=1, col=ring_column)
    fig.update_layout(
        title=f'Cuts of {getattr(data, "filename", None) or "the image"}',
        template='plotly_white',
        height=450,
        width=430 * len(titles) + 150,
        coloraxis={
            'colorscale': INTENSITY_COLORSCALE,
            'cmin': limits[0] if log_scale else None,
            'cmax': limits[1] if log_scale else None,
            'colorbar': {'title': 'log₁₀ I' if log_scale else 'I', 'x': -0.12},
        },
    )
    if resolve_show(show):
        fig.show()
    return fig


def _outlines(cut: Any, record: dict | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed (qx, qy) paths around the region of *cut*, empty when unknown."""
    if record is None and hasattr(cut, 'bin_width'):  # an annular average
        return [_arc(radius, 0.0, 360.0) for radius in (cut.qmin, cut.qmax)]
    if not isinstance(record, dict):
        return []
    inner, outer = record['qmin'], record['qmax']
    if record['kind'] == 'circular':
        return [_arc(radius, 0.0, 360.0) for radius in (inner, outer)]
    centre, half_width = record['phi'], record['half_width']
    centres = [centre, centre + 180.0] if record['mirror'] else [centre]
    paths = []
    for middle in centres:
        start, stop = middle - half_width, middle + half_width
        x_out, y_out = _arc(outer, start, stop)
        x_in, y_in = _arc(inner, stop, start)
        paths.append(
            (np.concatenate([x_out, x_in, x_out[:1]]), np.concatenate([y_out, y_in, y_out[:1]]))
        )
    return paths


def _arc(radius: float, start: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    angles = np.radians(np.linspace(start, stop, 121))
    return radius * np.cos(angles), radius * np.sin(angles)


def plot_annular(ring: Any, fit: Any = None, show: bool | None = None) -> go.Figure:
    """I(φ) of an annular average, with the ``fit_sin2`` curve when given."""
    from .plotting import error_bars, resolve_show

    fig = go.Figure(
        go.Scatter(
            x=ring.phi,
            y=ring.intensity,
            error_y=error_bars(ring.uncertainty),
            mode='markers',
            name='data',
        )
    )
    title = ring.title
    if fit is not None:
        phi = np.linspace(0.0, 360.0, 361)
        fig.add_trace(
            go.Scatter(x=phi, y=fit.evaluate(phi), mode='lines', name='A + B sin²(φ − φ₀)')
        )
        title += f': φ₀ = {fit.phi0:.1f}°, B/A = {fit.ratio:.3g}'
    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis={'title': 'φ (degrees)', 'dtick': 45},
        yaxis={'title': 'I(φ)'},
        height=420,
        width=700,
    )
    if resolve_show(show):
        fig.show()
    return fig
