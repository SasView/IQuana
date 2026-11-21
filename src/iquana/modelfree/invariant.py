"""The scattering invariant Q* = ∫q²(I − B)dq, with volume fraction and specific surface."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import erf

from .common import (
    check_delta_sld,
    check_finite,
    check_volume_fraction,
    check_window,
    finish,
    prepare,
    require_points,
    require_same_background,
    require_same_data,
)
from .result import (
    ABSOLUTE_SCALE_FACTOR,
    GuinierResult,
    InvariantResult,
    ModelFreeError,
    ModelFreeFit,
    PorodResult,
    PowerLawResult,
)

EXTRAPOLATION_LOW_LIMIT = 0.05
EXTRAPOLATION_HIGH_LIMIT = 0.10
MAX_PHI_PRODUCT = 0.25  # φ(1−φ) cannot exceed 1/4
CURVE_POINTS = 60
HIGH_Q_CURVE_SPAN = 4.0  # the drawn high-Q tail runs to this multiple of q₂
GRADIENT_STEP = 1e-6


def invariant(
    data: Any,
    low_q: GuinierResult | PowerLawResult | None = None,
    high_q: PorodResult | PowerLawResult | None = None,
    background: float | None = None,
    delta_sld: float | None = None,
    volume_fraction: float | None = None,
    qmin: float | None = None,
    qmax: float | None = None,
) -> InvariantResult:
    """Integrate Q* = ∫₀^∞ q²·(I − B) dq: measured part plus two extrapolations.

    The measured part is a trapezoid from q₁ to q₂, the first and last
    measured points in [qmin, qmax]. Below q₁ a ``'standard'`` Guinier result
    or a power law with n < 3 extrapolates; above q₂ a Porod result (Kp/q₂) or
    a power law with n > 3. A missing extrapolation truncates Q*.

    One background B applies to all three parts: ``background``, or else the
    one recorded by ``high_q``, then ``low_q``, else 0. Results computed with a
    different background raise :class:`ModelFreeError`; fit Porod first and
    pass its ``background`` to :func:`guinier`.

    The uncertainty is propagated once from the dataset's intensities, through
    the trapezoid and through each fit's sensitivity to the same points, so the
    shared data and a fitted background are counted once. It assumes
    independent Gaussian errors and first-order propagation; ``delta_sld`` and
    ``volume_fraction`` are treated as exact.

    Args:
        delta_sld: SLD difference in 10⁻⁶ Å⁻² (``2.5`` for 2.5 × 10⁻⁶ Å⁻²).
            Gives φ(1−φ) = Q* × 10⁴/(2π²·d²) and, with a Porod result,
            S/V = Kp × 10⁴/(2π·d²). Both need intensity in cm⁻¹.
        volume_fraction: The known volume fraction φ. With a Porod result
            gives S/V = π·φ(1−φ)·Kp/Q*, which needs neither absolute scale nor
            contrast.

    Raises:
        ModelFreeError: For a diverging extrapolation (low-Q n ≥ 3, high-Q
            n ≤ 3), a rod or sheet Guinier result, mismatched backgrounds or
            datasets, Q* ≤ 0, or φ(1−φ) > 1/4.
    """
    check_window(qmin, qmax)
    if background is not None:
        background = check_finite('background', background)
    d = None if delta_sld is None else check_delta_sld(delta_sld)
    phi_known = None if volume_fraction is None else check_volume_fraction(volume_fraction)
    _check_inputs(low_q, high_q)

    points = prepare(
        data, 'Invariant', qmin, qmax, fabricated_note='The uncertainty of Q* is not meaningful.'
    )
    require_points(points.q.size, 2, 'The invariant')
    require_same_data(points, low_q, high_q)
    fits = [r for r in (high_q, low_q) if r is not None]
    if background is None:
        background = fits[0].background if fits else 0.0
    for r in fits:
        require_same_background(background, r, f'the {type(r).__name__} fit')

    q, n_total = points.q, points.n_total
    q1, q2 = float(q[0]), float(q[-1])
    sigma = np.zeros(n_total)
    for r in (points, *fits):
        sigma[r.index] = r.sigma
    source = next((r for r in fits if _background_fitted(r)), None)
    d_background = np.zeros(n_total)  # dB/dI over the dataset
    if source is not None:
        d_background[source.index] = source.sensitivity[-1]

    # Measured part: trapezoid of q²(I − B), and its gradient over the dataset.
    widths = np.diff(q)
    weights = q**2 * (np.append(widths, 0.0) + np.insert(widths, 0, 0.0)) / 2.0
    measured = float(weights @ (points.intensity - background))
    g_measured = -weights.sum() * d_background
    g_measured[points.index] += weights

    low, g_low, low_method, low_curve = _extrapolate(
        low_q, q1, 'low', n_total, source, d_background
    )
    high, g_high, high_method, high_curve = _extrapolate(
        high_q, q2, 'high', n_total, source, d_background
    )
    qstar = low + measured + high
    if qstar <= 0:
        raise ModelFreeError(
            f'Q* = {qstar:.3g} <= 0: the background ({background:.3g}) is probably too large.'
        )
    g_qstar = g_low + g_measured + g_high

    def err(gradient: np.ndarray) -> float:
        return float(np.sqrt(np.sum((gradient * sigma) ** 2)))

    result = InvariantResult(
        **points.result_fields(background),
        qstar=qstar,
        qstar_err=err(g_qstar),
        low_q_part=low,
        low_q_part_err=err(g_low),
        measured_part=measured,
        measured_part_err=err(g_measured),
        high_q_part=high,
        high_q_part_err=err(g_high),
        low_q_method=low_method,
        high_q_method=high_method,
        low_q_curve=low_curve,
        high_q_curve=high_curve,
        delta_sld=d,
        known_volume_fraction=phi_known,
    )

    porod_gradient = None
    if isinstance(high_q, PorodResult):
        porod_gradient = _theta_gradient(high_q, n_total, source, d_background)[0]
    if d is not None:
        factor = ABSOLUTE_SCALE_FACTOR / (2.0 * math.pi**2 * d**2)
        phi_product = qstar * factor
        if phi_product > MAX_PHI_PRODUCT:
            raise ModelFreeError(
                f'phi(1-phi) = {phi_product:.3g} exceeds 1/4: the intensity is not on '
                'absolute scale (cm^-1), or delta_sld is wrong.'
            )
        root = math.sqrt(1.0 - 4.0 * phi_product)
        result.phi_product, result.phi_product_err = phi_product, err(factor * g_qstar)
        result.volume_fraction = (1.0 - root) / 2.0
        result.volume_fraction_complement = (1.0 + root) / 2.0
        result.volume_fraction_err = err(factor * g_qstar / root) if root > 0 else math.inf
        if porod_gradient is not None:
            sv_factor = ABSOLUTE_SCALE_FACTOR / (2.0 * math.pi * d**2)
            result.specific_surface_porod = high_q.porod_constant * sv_factor
            result.specific_surface_porod_err = err(sv_factor * porod_gradient)
    if phi_known is not None and porod_gradient is not None:
        product = math.pi * phi_known * (1.0 - phi_known)
        kp = high_q.porod_constant
        result.specific_surface_invariant = product * kp / qstar
        result.specific_surface_invariant_err = err(
            product * (porod_gradient / qstar - kp * g_qstar / qstar**2)
        )
    if d is not None and phi_known is not None:
        result.scale_check = result.phi_product / (phi_known * (1.0 - phi_known))

    _check(result, low_q, high_q, source)
    return finish(result)


def _check_inputs(low_q, high_q) -> None:
    if low_q is not None and not isinstance(low_q, (GuinierResult, PowerLawResult)):
        raise TypeError(
            f'low_q must be a GuinierResult or a PowerLawResult, got {type(low_q).__name__}.'
        )
    if high_q is not None and not isinstance(high_q, (PorodResult, PowerLawResult)):
        raise TypeError(
            f'high_q must be a PorodResult or a PowerLawResult, got {type(high_q).__name__}.'
        )
    if isinstance(low_q, GuinierResult) and low_q.kind != 'standard':
        raise ModelFreeError(
            f"The low-Q extrapolation needs a kind='standard' Guinier result, not {low_q.kind!r}."
        )
    if isinstance(low_q, PowerLawResult) and low_q.exponent >= 3:
        raise ModelFreeError(
            f'A low-Q power law with n = {low_q.exponent:.3g} >= 3 makes the invariant diverge.'
        )
    if isinstance(high_q, PowerLawResult) and high_q.exponent <= 3:
        raise ModelFreeError(
            f'A high-Q power law with n = {high_q.exponent:.3g} <= 3 makes the invariant diverge.'
        )


def _background_fitted(result: ModelFreeFit) -> bool:
    return isinstance(result, PorodResult) or getattr(result, 'background_fitted', False)


def _theta_gradient(result: ModelFreeFit, n_total, source, d_background) -> np.ndarray:
    """d(parameters)/d(intensity) over the whole dataset, a fixed background chained to its fit."""
    gradient = np.zeros((result.theta.size, n_total))
    gradient[:, result.index] = result.sensitivity
    if result is not source and result.background_sensitivity is not None:
        gradient += np.outer(result.background_sensitivity, d_background)
    return gradient


def _tail(result: ModelFreeFit, edge: float, side: str) -> tuple[Callable, str, np.ndarray]:
    """(part(theta), method, curve q grid) for one extrapolation."""
    if isinstance(result, GuinierResult):

        def part(theta):
            rg, i0 = theta[0], theta[1]
            a = rg**2 / 3.0
            return i0 * (
                math.sqrt(math.pi) * erf(math.sqrt(a) * edge) / (4.0 * a**1.5)
                - edge * math.exp(-a * edge**2) / (2.0 * a)
            )

        return part, 'Guinier', np.linspace(edge / CURVE_POINTS, edge, CURVE_POINTS)
    if isinstance(result, PorodResult):
        return (
            lambda theta: theta[0] / edge,
            'Porod',
            np.geomspace(edge, HIGH_Q_CURVE_SPAN * edge, CURVE_POINTS),
        )
    sign = 1.0 if side == 'low' else -1.0

    def part(theta):
        a, n = theta[0], theta[1]
        return sign * a * edge ** (3.0 - n) / (3.0 - n)

    grid = (
        np.linspace(edge / CURVE_POINTS, edge, CURVE_POINTS)
        if side == 'low'
        else np.geomspace(edge, HIGH_Q_CURVE_SPAN * edge, CURVE_POINTS)
    )
    return part, 'power law', grid


def _extrapolate(result, edge, side, n_total, source, d_background):
    """(value, gradient over the dataset, method, curve) of one extrapolated part."""
    if result is None:
        return 0.0, np.zeros(n_total), None, None
    part, method, grid = _tail(result, edge, side)
    theta = result.theta
    grad = np.zeros(theta.size)
    for j in range(theta.size):
        step = GRADIENT_STEP * max(abs(theta[j]), 1e-12)
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        grad[j] = (part(up) - part(down)) / (2.0 * step)
    curve = np.vstack([grid, grid**2 * (result.evaluate(grid) - result.background)])
    gradient = grad @ _theta_gradient(result, n_total, source, d_background)
    return float(part(theta)), gradient, method, curve


def _check(result: InvariantResult, low_q, high_q, source) -> None:
    add = result.warnings.append
    q1, q2 = result.qmin, result.qmax
    for fit, edge, side in ((low_q, q1, 'low'), (high_q, q2, 'high')):
        if fit is None:
            add(
                (
                    'truncated',
                    f'No {side}-Q extrapolation: Q* is truncated at q = {edge:.4g}.',
                )
            )
        elif not fit.qmin <= edge <= fit.qmax:
            hint = (
                f' The measured part still contains the points below {fit.qmin:.4g} (an '
                f'upturn cut from the fit?): pass qmin={fit.qmin:.6g} to invariant().'
                if side == 'low' and fit.qmin > edge
                else ''
            )
            add(
                (
                    'window_not_at_edge',
                    f'The {side}-Q fit covers {fit.qmin:.4g} to {fit.qmax:.4g} but extrapolates '
                    f'from q = {edge:.4g}.{hint}',
                )
            )
    if result.low_q_fraction > EXTRAPOLATION_LOW_LIMIT:
        add(
            (
                'extrapolation_low',
                f'The low-Q extrapolation is {100 * result.low_q_fraction:.1f} % of Q* '
                f'(above {100 * EXTRAPOLATION_LOW_LIMIT:g} %).',
            )
        )
    if result.high_q_fraction > EXTRAPOLATION_HIGH_LIMIT:
        add(
            (
                'extrapolation_high',
                f'The high-Q extrapolation is {100 * result.high_q_fraction:.1f} % of Q* '
                f'(above {100 * EXTRAPOLATION_HIGH_LIMIT:g} %).',
            )
        )
    if source is None and result.background != 0:
        add(
            (
                'background_uncertainty_excluded',
                f'The background {result.background:.6g} was not fitted here, so its '
                'uncertainty is not included in the error of Q*.',
            )
        )
