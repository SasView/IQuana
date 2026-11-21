"""Porod analysis: I = Kp·q⁻⁴ + B at high Q."""

import math
from typing import Any

import numpy as np

from .common import check_window, finish, prepare, require_points
from .linear import weighted_polyfit
from .power_law import _fit_free
from .result import ModelFreeError, PorodResult

EXPONENT_TOLERANCE = 0.3
EXPONENT_SIGMAS = 3.0


def porod(data: Any, qmin: float, qmax: float | None = None) -> PorodResult:
    """Fit I = Kp·q⁻⁴ + B over [qmin, qmax]: the Porod constant and the flat background.

    Fitted as a weighted line of q⁴·I against q⁴ (intercept Kp, slope B), which
    is the same weighted least squares as fitting I directly. An independent
    three-parameter power law over the same window checks the exponent: the
    ``porod_exponent`` warning fires when it differs from 4 by more than 0.3
    and by more than 3σ (below 4: a rough or fractal surface; above 4: a
    diffuse interface). Use ``result.background`` as the background of the
    other analyses, so all of them subtract the same level.

    Raises:
        InsufficientDataError: Fewer than 3 usable points.
        ModelFreeError: When the fitted Porod constant is negative.
    """
    if qmin is None:
        raise ValueError('porod() needs qmin, the start of the Porod regime.')
    check_window(qmin, qmax)

    points = prepare(data, 'Porod analysis', qmin, qmax)
    require_points(points.q.size, 3, 'Porod analysis')
    q4 = points.q**4
    line = weighted_polyfit(q4, q4 * points.intensity, q4 * points.sigma, 1)
    porod_constant, background = (float(v) for v in line.coef)
    if porod_constant < 0:
        raise ModelFreeError(
            f'The fitted Porod constant is negative ({porod_constant:.3g}): the window '
            'holds no q^-4 decay. Move qmin to higher Q, or check the background.'
        )
    dof = points.q.size - 2
    cov = line.cov * (line.chisq / dof if points.fabricated and dof > 0 else 1.0)
    fit = porod_constant / q4 + background

    result = PorodResult(
        **points.result_fields(background),
        fit=fit,
        chisq=float(np.sum(((points.intensity - fit) / points.sigma) ** 2)),
        chisq_linearised=line.chisq,
        cov=cov,
        sensitivity=line.hat * q4,
        porod_constant=porod_constant,
        porod_constant_err=math.sqrt(cov[0, 0]),
        background_err=math.sqrt(cov[1, 1]),
    )
    _check_exponent(result, points)
    return finish(result)


def _check_exponent(result: PorodResult, points) -> None:
    """Fill ``free_exponent`` from a three-parameter power law; warn if it is not 4."""
    if points.q.size < 4:
        return
    try:
        free = _fit_free(points, result.background)
    except (ModelFreeError, np.linalg.LinAlgError):
        return
    result.free_exponent, result.free_exponent_err = free.exponent, free.exponent_err
    deviation = abs(free.exponent - 4.0)
    if deviation > EXPONENT_TOLERANCE and deviation > EXPONENT_SIGMAS * free.exponent_err:
        meaning = 'a rough or fractal surface' if free.exponent < 4 else 'a diffuse interface'
        result.warnings.append(
            (
                'porod_exponent',
                f'A free power law over the Porod window gives an exponent of '
                f'{free.exponent:.3f} +/- {free.exponent_err:.2g}, not 4: {meaning}, or a '
                'window outside the Porod regime.',
            )
        )
