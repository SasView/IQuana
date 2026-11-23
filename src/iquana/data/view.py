"""Dimension-agnostic access to a dataset's arrays.

1D data (``Data1D``) keeps its intensity in ``y``/``dy`` and its Q in ``x``;
2D data (``Data2D``) keeps them in ``data``/``err_data`` and
``qx_data``/``qy_data``. Generic code asks these functions instead of reading
the attributes, so the 1D/2D distinction lives in one place.

SESANS data is a ``Data1D`` too, but its ``x`` is a spin-echo length and its
``y`` a normalised depolarisation. :func:`is_sesans` tells it apart and
:func:`axis_text` gives the words and axes that describe each kind.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..console import ANGSTROM, INVERSE_ANGSTROM


def is_2d(data: Any) -> bool:
    """True for 2D detector data (sasdata or sasmodels ``Data2D``)."""
    return data is not None and getattr(data, 'qx_data', None) is not None


def is_sesans(data: Any) -> bool:
    """True for SESANS (or reduced SEMSANS) data: depolarisation against spin-echo length."""
    return bool(getattr(data, 'isSesans', False))


@dataclass(frozen=True, slots=True)
class AxisText:
    """How a kind of data is named in messages, plots and reports."""

    #: 'SANS' or 'SESANS', for titles.
    measurement: str
    #: What the x values are, for messages ('Q range for fitting').
    x_name: str
    #: The unit of x in console text (plain ASCII where the console needs it).
    x_unit: str
    #: Plot axis titles.
    x_title: str
    y_title: str
    #: Whether the data reads best on log axes.
    log_axes: bool


SANS_AXES = AxisText('SANS', 'Q', INVERSE_ANGSTROM, 'Q (Å⁻¹)', 'I(Q)', True)
#: Linear axes: the depolarisation is negative, so a log axis cannot show it.
SESANS_AXES = AxisText(
    'SESANS',
    'Spin-echo length',
    ANGSTROM,
    'Spin-echo length (Å)',
    'ln(P/P₀)/(tλ²) (Å⁻² cm⁻¹)',
    False,
)


def axis_text(data: Any) -> AxisText:
    """The words and axes for *data*; I(Q) when there is no data."""
    return SESANS_AXES if is_sesans(data) else SANS_AXES


def intensity(data: Any) -> np.ndarray:
    """The measured intensity column."""
    return np.asarray(data.data if is_2d(data) else data.y, dtype=float)


def uncertainty(data: Any) -> np.ndarray | None:
    """The intensity uncertainty column, or None when the dataset has none."""
    values = data.err_data if is_2d(data) else data.dy
    return None if values is None else np.asarray(values, dtype=float)


def set_uncertainty(data: Any, sigma: np.ndarray) -> None:
    """Write *sigma* into the column sasmodels reads as the uncertainty."""
    if is_2d(data):
        data.err_data = sigma
    else:
        data.dy = sigma


def q_magnitude(data: Any) -> np.ndarray:
    """|Q| per point. For 2D data computed from qx/qy, exactly as sasmodels does."""
    if is_2d(data):
        return np.hypot(
            np.asarray(data.qx_data, dtype=float), np.asarray(data.qy_data, dtype=float)
        )
    return np.asarray(data.x, dtype=float)


def n_points(data: Any) -> int:
    """Number of points (pixels for 2D data), counted from the coordinates."""
    return int(q_magnitude(data).size)
