"""Validation of a 1D dataset for the model-free analyses (P(r), Guinier, Porod, ...).

Shared by :mod:`iquana.inversion` and :mod:`iquana.modelfree`, so both accept and
reject the same datasets. Nothing here mutates the dataset.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .loader import has_real_data

FABRICATED_SIGMA_RELATIVE = 0.05
FABRICATED_SIGMA_FLOOR_FRACTION = 0.01
FABRICATED_SIGMA_FORMULA = (
    f'sigma = max({FABRICATED_SIGMA_RELATIVE}*|I|, {FABRICATED_SIGMA_FLOOR_FRACTION}*median|I|)'
)
PR_FABRICATED_NOTE = 'Chi-squared-based diagnostics are not interpretable.'
SLIT_SMEARING_POLICIES = ('warn', 'raise')


class InsufficientDataError(ValueError):
    """The dataset does not carry enough usable points for the request."""


@dataclass(slots=True)
class _PreparedData:
    """Validated arrays plus the full-length accepted mask, built once per call."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    accepted: np.ndarray
    uncertainties_fabricated: bool
    n_dropped_points: int


def prepare_1d(
    data: Any,
    purpose: str = 'P(r) inversion',
    qmin: float | None = None,
    qmax: float | None = None,
    slit_smearing: str = 'warn',
    fabricated_note: str | None = PR_FABRICATED_NOTE,
    stacklevel: int = 3,
) -> _PreparedData:
    """Validate a dataset and build the accepted-point arrays without mutating it.

    Accepts a general ``Data1D``, normalized (with ``qmin``/``qmax``/``mask``)
    or raw. For raw data no q-cut is invented: all finite, unmasked points are
    used. Points with non-finite q, I or sigma and ``q == 0`` points are
    dropped with a warning.

    Args:
        data: The dataset.
        purpose: Name of the calculation, used in error messages.
        qmin, qmax: Replace the dataset's own Q limit on that side when given.
        slit_smearing: ``'warn'`` or ``'raise'`` when the dataset carries real
            slit-smearing columns (dxl/dxw).
        fabricated_note: Closing sentence of the warning issued when dI is
            missing. None suppresses the warning; the caller reports it.
        stacklevel: Stack level of the warnings, so they point at the caller's
            caller.
    """
    if getattr(data, 'qx_data', None) is not None:
        raise TypeError(f'{purpose} supports 1D data only, got 2D data.')
    if getattr(data, 'isSesans', False):
        raise TypeError(f'{purpose} does not support SESANS data.')

    x_raw = getattr(data, 'x', None)
    y_raw = getattr(data, 'y', None)
    if x_raw is None or y_raw is None:
        raise ValueError('Dataset has no data: x and y must be populated.')
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError('Dataset has no data: x and y must be populated.')
    if x.size != y.size:
        raise ValueError(f'Length mismatch: x has {x.size} points but y has {y.size}.')

    dy_raw = getattr(data, 'dy', None)
    has_dy = has_real_data(dy_raw)
    if has_dy and np.asarray(dy_raw).size != x.size:
        raise ValueError(
            f'Length mismatch: x has {x.size} points but dy has {np.asarray(dy_raw).size}.'
        )

    if np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Dataset contains negative q values; q must be non-negative.')

    if slit_smearing not in SLIT_SMEARING_POLICIES:
        raise ValueError(f'slit_smearing must be one of {SLIT_SMEARING_POLICIES}.')
    if any(has_real_data(getattr(data, attr, None)) for attr in ('dxl', 'dxw')):
        if slit_smearing == 'raise':
            raise ValueError(
                f'{purpose} does not support slit-smeared data (the dataset carries '
                'dxl/dxw columns): the result would be wrong in kind, not just '
                'blurred. Desmear the data first.'
            )
        warnings.warn(
            'Dataset carries slit-smearing resolution columns (dxl/dxw); '
            f'{purpose} does not support smearing and will treat the '
            'data as unsmeared. Results for USANS-type data will be wrong.',
            stacklevel=stacklevel,
        )

    # Same rule as the fit engines (get_fit_index): inside [lower, upper],
    # unmasked, intensity not NaN. An explicit limit replaces the dataset's.
    lower = qmin if qmin is not None else getattr(data, 'qmin', None)
    upper = qmax if qmax is not None else getattr(data, 'qmax', None)
    base = ~np.isnan(y)
    if lower is not None:
        base &= x >= lower
    if upper is not None:
        base &= x <= upper
    mask = getattr(data, 'mask', None)
    if mask is not None and np.asarray(mask).size == y.size:
        base &= ~np.asarray(mask, dtype=bool)

    if has_dy:
        sigma = np.asarray(dy_raw, dtype=float)
        fabricated = False
    else:
        finite_i = np.abs(y[np.isfinite(y)])
        median_scale = float(np.median(finite_i)) if finite_i.size else 0.0
        sigma = np.maximum(
            FABRICATED_SIGMA_RELATIVE * np.abs(y),
            FABRICATED_SIGMA_FLOOR_FRACTION * median_scale,
        )
        fabricated = True
        if fabricated_note is not None:
            warnings.warn(
                'Dataset has no intensity uncertainties (dI); fabricated '
                f'{FABRICATED_SIGMA_FORMULA} will be used. {fabricated_note}',
                stacklevel=stacklevel,
            )

    accepted = base & np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & (x > 0)
    n_dropped = int(base.sum()) - int(accepted.sum())
    if n_dropped > 0:
        warnings.warn(
            f'{n_dropped} point(s) dropped from the {purpose} (non-finite q/I/sigma, '
            'non-positive sigma, or q == 0).',
            stacklevel=stacklevel,
        )
    if not accepted.any():
        raise InsufficientDataError('No usable data points remain after validation.')

    return _PreparedData(
        q=x[accepted],
        intensity=y[accepted],
        sigma=sigma[accepted],
        accepted=accepted,
        uncertainties_fabricated=fabricated,
        n_dropped_points=n_dropped,
    )
