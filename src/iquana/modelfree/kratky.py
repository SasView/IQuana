"""Kratky transform: q²·I against q, or its dimensionless form."""

from typing import Any

import numpy as np

from .common import (
    check_finite,
    finish,
    prepare,
    require_points,
    require_same_background,
    require_same_data,
)
from .result import GuinierResult, KratkyCurve, ModelFreeError

DIMENSIONLESS_PEAK_SEARCH = 5.0  # q·Rg; beyond it noise and background errors dominate


def kratky(
    data: Any, guinier: GuinierResult | None = None, background: float | None = None
) -> KratkyCurve:
    """Kratky curve of a 1D dataset.

    Without ``guinier``: q²·(I − B) against q, with B = ``background`` or 0.
    With a ``'standard'`` Guinier result: the dimensionless form
    (q·Rg)²·(I − B)/I(0) against q·Rg, with B the Guinier result's background.

    The dimensionless plot marks q·Rg = √3 and height 3/e ≈ 1.104, the peak of
    the Guinier law x²·exp(−x²/3) and the usual reference for globular
    proteins. It is not the peak of a real compact particle: a homogeneous
    sphere peaks at q·Rg = 1.612 with height 1.027. No verdict ("folded",
    "flexible") is given.

    The peak is searched over q·Rg ≤ 5 in the dimensionless form, where the
    peak of a particle lies (beyond it (q·Rg)² amplifies noise and background
    errors), and over the whole curve otherwise. A maximum at the edge of the
    searched range is no peak: the curve is still rising, and the peak is None.

    Raises:
        ModelFreeError: For a rod or sheet Guinier result, a result from another
            dataset, or a ``background`` different from the Guinier result's.
    """
    if background is not None:
        background = check_finite('background', background)
    points = prepare(data, 'Kratky analysis', fabricated_note='The error bars are not measured.')
    require_points(points.q.size, 3, 'A Kratky curve')

    if guinier is not None:
        if guinier.kind != 'standard':
            raise ModelFreeError(
                f"The dimensionless Kratky form needs a kind='standard' Guinier result, "
                f'not {guinier.kind!r}.'
            )
        require_same_data(points, guinier)
        if background is not None:
            require_same_background(background, guinier, 'kratky()')
        background = guinier.background
    elif background is None:
        background = 0.0

    excess = points.intensity - background
    if guinier is None:
        x, scale = points.q, points.q**2
    else:
        x = points.q * guinier.rg
        scale = x**2 / guinier.i0
    y, dy = scale * excess, scale * points.sigma
    searched = x <= DIMENSIONLESS_PEAK_SEARCH if guinier is not None else np.ones(x.size, bool)
    position, height = _peak(x[searched], y[searched])
    return finish(
        KratkyCurve(
            **points.result_fields(background),
            x=x,
            y=y,
            dy=dy,
            dimensionless=guinier is not None,
            peak_position=position,
            peak_height=height,
        )
    )


def _peak(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Maximum of the curve, refined by a parabola through the top three points."""
    if y.size < 3:
        return None, None
    top = int(np.argmax(y))
    if top == 0 or top == y.size - 1:
        return None, None
    coef = np.polyfit(x[top - 1 : top + 2], y[top - 1 : top + 2], 2)
    if coef[0] >= 0:
        return float(x[top]), float(y[top])
    vertex = -coef[1] / (2.0 * coef[0])
    return float(vertex), float(np.polyval(coef, vertex))
