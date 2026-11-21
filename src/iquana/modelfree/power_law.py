"""Power law I − B = A·q⁻ⁿ over a chosen Q window."""

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .common import Points, check_finite, check_window, finish, prepare, require_points
from .linear import weighted_polyfit
from .result import ModelFreeError, PowerLawResult

SHORT_RANGE_RATIO = 2.0  # window shorter than a factor of 2 in Q: warn
LOW_SIGNAL_DROPPED_FRACTION = 0.05
LOW_SIGNAL_MEDIAN_SNR = 3.0
CONDITION_LIMIT = 1e10  # column-normalised Jacobian; above this, no covariance


def power_law(
    data: Any,
    qmin: float,
    qmax: float,
    background: float = 0.0,
    fit_background: bool = False,
) -> PowerLawResult:
    """Fit I − B = A·q⁻ⁿ over [qmin, qmax].

    With a fixed background (the default) this is a weighted line of ln(I − B)
    against ln q; points with I − B ≤ 0 are dropped and counted. Near the
    background level that log fit is biased (``low_signal`` warning): use
    ``fit_background=True``, a three-parameter fit in linear space with
    ``scipy.optimize.least_squares``, where ``background`` is the starting value
    of B. A is kept positive; n and B are free.

    Raises:
        InsufficientDataError: Fewer than 3 usable points (4 with
            ``fit_background=True``).
        ModelFreeError: When the nonlinear fit does not converge or its
            parameters are not determined by the data.
    """
    if qmin is None or qmax is None:
        raise ValueError('power_law() needs both qmin and qmax.')
    check_window(qmin, qmax)
    background = check_finite('background', background)

    points = prepare(data, 'Power-law analysis', qmin, qmax)
    if fit_background:
        require_points(points.q.size, 4, 'A power law with a fitted background')
        result = _fit_free(points, background)
    else:
        result = _fit_fixed(points, background)

    if result.qmax / result.qmin < SHORT_RANGE_RATIO:
        result.warnings.append(
            (
                'short_q_range',
                f'The window spans only a factor of {result.qmax / result.qmin:.3g} in Q '
                f'(less than {SHORT_RANGE_RATIO:g}); an exponent from it means little.',
            )
        )
    return finish(result)


def _fit_fixed(points: Points, background: float) -> PowerLawResult:
    """Weighted line of ln(I − B) against ln q, B fixed."""
    excess = points.intensity - background
    positive = excess > 0
    n_nonpositive = int(np.count_nonzero(~positive))
    snr = float(np.median(excess / points.sigma))
    if n_nonpositive > LOW_SIGNAL_DROPPED_FRACTION * excess.size or snr < LOW_SIGNAL_MEDIAN_SNR:
        points.warnings.append(
            (
                'low_signal',
                f'The window is near the background level ({n_nonpositive} of {excess.size} '
                f'points with I - B <= 0, median (I - B)/sigma = {snr:.3g}); the log fit is '
                'biased there. Use fit_background=True.',
            )
        )
    points = points.take(positive)
    excess = excess[positive]
    require_points(points.q.size, 3, 'A power law with a fixed background')

    line = weighted_polyfit(np.log(points.q), np.log(excess), points.sigma / excess, 1)
    prefactor, exponent = math.exp(line.coef[0]), -line.coef[1]
    jac = np.array([[prefactor, 0.0], [0.0, -1.0]])  # d(A, n)/d(ln A, slope)
    dof = points.q.size - 2
    cov = jac @ (line.cov * (line.chisq / dof if points.fabricated and dof > 0 else 1.0)) @ jac.T
    fit = prefactor * points.q ** (-exponent) + background
    return PowerLawResult(
        **points.result_fields(background, n_nonpositive),
        fit=fit,
        chisq=float(np.sum(((points.intensity - fit) / points.sigma) ** 2)),
        chisq_linearised=line.chisq,
        cov=cov,
        sensitivity=np.vstack([jac @ line.hat / excess, np.zeros(points.q.size)]),
        background_sensitivity=np.append(jac @ line.hat @ (-1.0 / excess), 1.0),
        exponent=exponent,
        exponent_err=math.sqrt(cov[1, 1]),
        prefactor=prefactor,
        prefactor_err=math.sqrt(cov[0, 0]),
    )


def _fit_free(points: Points, background: float) -> PowerLawResult:
    """Three-parameter fit of A·q⁻ⁿ + B in linear space, started at the fixed-B line."""
    q, intensity, sigma = points.q, points.intensity, points.sigma
    excess = intensity - background
    positive = excess > 0
    if np.count_nonzero(positive) >= 3:
        start = weighted_polyfit(
            np.log(q[positive]), np.log(excess[positive]), sigma[positive] / excess[positive], 1
        ).coef
        x0 = [math.exp(start[0]), -start[1], background]
    else:
        x0 = [
            max(float(np.median(np.abs(excess))), 1e-30) * float(np.median(q)) ** 4,
            4.0,
            background,
        ]

    def model_jacobian(theta):
        a, n, _ = theta
        power = q ** (-n)
        return np.column_stack([power, -a * np.log(q) * power, np.ones_like(q)]) / sigma[:, None]

    def residuals(theta):
        a, n, b = theta
        return (a * q ** (-n) + b - intensity) / sigma

    fit = least_squares(
        residuals,
        x0,
        jac=model_jacobian,
        bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
        x_scale='jac',
    )
    jac = model_jacobian(fit.x)
    norms = np.linalg.norm(jac, axis=0)
    if (
        not fit.success
        or fit.x[0] <= 0
        or np.any(norms == 0)
        or np.linalg.cond(jac / norms) > CONDITION_LIMIT
    ):
        raise ModelFreeError(
            'The power-law fit with a fitted background failed: '
            f'{fit.message if not fit.success else "the parameters are not determined by the data"}. '
            'The window may hold no power law above the background.'
        )
    cov = np.linalg.inv(jac.T @ jac)
    dof = q.size - 3
    if points.fabricated and dof > 0:
        cov = cov * float(fit.fun @ fit.fun) / dof
    prefactor, exponent, b = (float(v) for v in fit.x)
    # The power-law term at the window's geometric-mean Q, where it is least
    # correlated with the exponent (A itself is the value at q = 1/Å).
    q_mid = math.exp(float(np.mean(np.log(q))))
    term = prefactor * q_mid ** (-exponent)
    grad = np.array([term / prefactor, -term * math.log(q_mid), 0.0])
    term_err = math.sqrt(float(grad @ cov @ grad))
    if term_err > term:
        raise ModelFreeError(
            f'The power-law fit with a fitted background cannot separate the power-law '
            f'term at q = {q_mid:.3g} ({term:.3g} +/- {term_err:.3g}) from zero: the window '
            'holds no power law above the background.'
        )
    model = prefactor * q ** (-exponent) + b
    chisq = float(fit.fun @ fit.fun)
    return PowerLawResult(
        **points.result_fields(b),
        fit=model,
        chisq=chisq,
        chisq_linearised=chisq,
        cov=cov,
        sensitivity=np.linalg.inv(jac.T @ jac) @ jac.T / sigma[None, :],
        method='least_squares (trf)',
        cov_source='jacobian',
        message=str(fit.message),
        exponent=exponent,
        exponent_err=math.sqrt(cov[1, 1]),
        prefactor=prefactor,
        prefactor_err=math.sqrt(cov[0, 0]),
        background_err=math.sqrt(cov[2, 2]),
        background_fitted=True,
    )
