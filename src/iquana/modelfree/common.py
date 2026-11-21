"""Shared plumbing of the model-free analyses: validation, data preparation, warnings."""

import hashlib
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..data.loader import has_real_data
from ..data.prepare import FABRICATED_SIGMA_FORMULA, InsufficientDataError, prepare_1d
from .result import ModelFreeError, ModelFreeResult

# delta_sld is in 10⁻⁶ Å⁻²; outside this magnitude it was almost surely typed in Å⁻².
DELTA_SLD_RANGE = (1e-3, 100.0)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def check_finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f'{name} must be a number, got {value!r}.') from None
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite, got {value}.')
    return number


def check_window(qmin: float | None, qmax: float | None) -> None:
    for name, value in (('qmin', qmin), ('qmax', qmax)):
        if value is not None and check_finite(name, value) <= 0:
            raise ValueError(f'{name} must be positive, got {value}.')
    if qmin is not None and qmax is not None and qmin >= qmax:
        raise ValueError(f'qmin ({qmin:g}) must be smaller than qmax ({qmax:g}).')


def check_delta_sld(delta_sld: float) -> float:
    d = check_finite('delta_sld', delta_sld)
    low, high = DELTA_SLD_RANGE
    if not low <= abs(d) <= high:
        raise ValueError(
            f'delta_sld = {d:g} is outside {low:g} to {high:g} in magnitude. It is the '
            'SLD difference in units of 1e-6 1/Ang^2, as every SLD in IQuana and '
            'sasmodels: pass 2.5 for 2.5e-6 1/Ang^2.'
        )
    return d


def check_volume_fraction(value: float) -> float:
    phi = check_finite('volume_fraction', value)
    if not 0.0 < phi < 1.0:
        raise ValueError(f'volume_fraction must lie strictly between 0 and 1, got {phi}.')
    return phi


def require_points(n: int, minimum: int, what: str) -> None:
    if n < minimum:
        raise InsufficientDataError(f'{what} needs at least {minimum} usable points, got {n}.')


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Points:
    """The accepted points of one call, sorted by Q, and what a result records about them."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    index: np.ndarray
    n_total: int
    digest: str
    fabricated: bool
    n_dropped: int
    warnings: list[tuple[str, str]] = field(default_factory=list)

    def take(self, keep: np.ndarray) -> 'Points':
        """A subset of the points; the warnings list is shared, not copied."""
        return replace(
            self,
            q=self.q[keep],
            intensity=self.intensity[keep],
            sigma=self.sigma[keep],
            index=self.index[keep],
        )

    def result_fields(self, background: float, n_nonpositive: int = 0) -> dict[str, Any]:
        """The :class:`ModelFreeResult` fields these points determine."""
        return {
            'qmin': float(self.q[0]),
            'qmax': float(self.q[-1]),
            'background': float(background),
            'data_digest': self.digest,
            'n_total': self.n_total,
            'index': self.index,
            'q': self.q,
            'intensity': self.intensity,
            'sigma': self.sigma,
            'warnings': self.warnings,
            'uncertainties_fabricated': self.fabricated,
            'n_dropped_points': self.n_dropped,
            'n_nonpositive_points': n_nonpositive,
        }


def data_digest(data: Any) -> str:
    """SHA-256 of the dataset's Q values, tying a result to its dataset."""
    return hashlib.sha256(np.ascontiguousarray(data.x, dtype=float).tobytes()).hexdigest()


def prepare(
    data: Any,
    purpose: str,
    qmin: float | None = None,
    qmax: float | None = None,
    fabricated_note: str = 'Parameter uncertainties are scaled by the reduced chi-squared.',
) -> Points:
    """Validate *data* and return its accepted points in [qmin, qmax], sorted by Q.

    Must be called directly from a public analysis function: warnings are
    raised with a stack level that points at that function's caller.
    """
    prep = prepare_1d(
        data,
        purpose=purpose,
        qmin=qmin,
        qmax=qmax,
        slit_smearing='raise',
        fabricated_note=None,
        stacklevel=4,
    )
    order = np.argsort(prep.q, kind='stable')
    points = Points(
        q=prep.q[order],
        intensity=prep.intensity[order],
        sigma=prep.sigma[order],
        index=np.flatnonzero(prep.accepted)[order],
        n_total=int(prep.accepted.size),
        digest=data_digest(data),
        fabricated=prep.uncertainties_fabricated,
        n_dropped=prep.n_dropped_points,
    )
    if points.fabricated:
        points.warnings.append(
            (
                'uncertainties_fabricated',
                f'Dataset has no intensity uncertainties (dI); fabricated '
                f'{FABRICATED_SIGMA_FORMULA} was used. {fabricated_note}',
            )
        )
    if has_real_data(getattr(data, 'dx', None)):
        points.warnings.append(
            ('dq_ignored', 'The dataset carries dQ resolution; model-free analysis ignores it.')
        )
    _note_outside_range(data, qmin, qmax, points)
    return points


def _note_outside_range(data: Any, qmin: float | None, qmax: float | None, points: Points) -> None:
    """Warn when an explicit window reaches past the dataset's own Q range."""
    x = np.asarray(data.x, dtype=float)
    low = getattr(data, 'qmin', None)
    high = getattr(data, 'qmax', None)
    low = float(np.nanmin(x)) if low is None else float(low)
    high = float(np.nanmax(x)) if high is None else float(high)
    outside = []
    if qmin is not None and qmin < low:
        outside.append(f'qmin = {qmin:g} is below the dataset range ({low:g})')
    if qmax is not None and qmax > high:
        outside.append(f'qmax = {qmax:g} is above the dataset range ({high:g})')
    if outside:
        points.warnings.append(
            (
                'outside_dataset_range',
                '; '.join(outside)
                + f'. The points used run from {points.q[0]:g} to {points.q[-1]:g}.',
            )
        )


# ---------------------------------------------------------------------------
# Consistency between chained results
# ---------------------------------------------------------------------------


def require_same_data(points: Points, *results: ModelFreeResult | None) -> None:
    for result in results:
        if result is not None and result.data_digest != points.digest:
            raise ModelFreeError(
                f'The {type(result).__name__} was computed on a different dataset '
                '(its Q values differ). Repeat it on this dataset.'
            )


def same_value(a: float, b: float) -> bool:
    return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def require_same_background(background: float, result: ModelFreeResult, call: str) -> None:
    if not same_value(background, result.background):
        raise ModelFreeError(
            f'Background mismatch: the {type(result).__name__} was computed with '
            f'background = {result.background:.6g}, but {background:.6g} is applied here. '
            f'One corrected curve needs one background: repeat {call} with '
            f'background={background:.6g}.'
        )


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def finish(result: ModelFreeResult) -> ModelFreeResult:
    """Emit the result's warnings at the caller of the public function, then return it."""
    for code, message in result.warnings:
        warnings.warn(f'{message} [{code}]', UserWarning, stacklevel=3)
    return result
