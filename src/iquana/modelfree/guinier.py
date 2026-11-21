"""Guinier analysis: Rg and I(0), or the cross-section of a rod or a sheet."""

import math
from typing import Any

import numpy as np

from .common import check_finite, check_window, finish, prepare, require_points
from .linear import LinearFit, weighted_polyfit
from .result import GUINIER_KINDS, GuinierKind, GuinierResult, ModelFreeError

LOW_Q_COVERAGE_LIMIT = 1.0  # q_min·Rg above this: the plateau is not reached
FEW_POINTS = 8  # below this, warn, and skip the curvature check
# Curvature c in ln I = ln I(0) − u/3 + c·u², u = (q·Rg)², windows to q·Rg = 1.3:
# sphere −0.009, Gaussian coil +0.025, thin rod +0.028, sphere with a
# hard-sphere S(q) −0.05 (φ = 0.02) to −0.30 (φ = 0.05).
CURVATURE_NEGATIVE = -0.03
CURVATURE_POSITIVE = 0.015
CURVATURE_SIGMAS = 3.0


def guinier(
    data: Any,
    qmin: float | None = None,
    qmax: float | None = None,
    kind: str = 'standard',
    qr_limit: float | None = None,
    background: float = 0.0,
    min_points: int = 5,
) -> GuinierResult:
    """Fit a Guinier law to the low-Q end of a 1D dataset.

    ========== ============================== ================ =================
    kind       law (B subtracted)             slope vs q²      result
    ========== ============================== ================ =================
    standard   ln I = ln I(0) − q²Rg²/3       −Rg²/3           Rg, I(0)
    rod        ln(q·I) = ln c − q²Rc²/2       −Rc²/2           cross-section Rc
    sheet      ln(q²·I) = ln c − q²Rt²        −Rt²             Rt, thickness √12·Rt
    ========== ============================== ================ =================

    ``'standard'`` is the law for the whole particle, whatever its shape.
    ``'rod'`` and ``'sheet'`` apply only between 2π/L and 1/R, so they need an
    explicit ``qmin`` above 2π/L.

    With no ``qmax`` the window grows from ``min_points`` points and the largest
    window with q_max·R ≤ ``qr_limit`` before the first violation is chosen.
    The low-Q edge is never trimmed automatically: cut an upturn with ``qmin``.

    Args:
        data: A 1D dataset (``fitter.data`` or a ``data_ops`` result).
        qmin, qmax: Window in Å⁻¹; each replaces the dataset's own limit.
        kind: ``'standard'``, ``'rod'`` or ``'sheet'``.
        qr_limit: Upper limit of q·R. Default 1.3 for ``'standard'`` (use 1.0
            for elongated or flexible particles) and 1.0 for the others.
        background: Flat level subtracted before the fit.
        min_points: Smallest window tried by the automatic scan.

    Raises:
        TypeError: For 2D or SESANS data.
        ValueError: For invalid arguments or slit-smeared data.
        InsufficientDataError: When fewer than ``min_points`` usable points remain.
        ModelFreeError: When no window satisfies the q·R limit, or the
            intensity does not fall with Q.
    """
    if kind not in GUINIER_KINDS:
        raise ValueError(f'kind must be one of {tuple(GUINIER_KINDS)}, got {kind!r}.')
    spec = GUINIER_KINDS[kind]
    check_window(qmin, qmax)
    background = check_finite('background', background)
    qr_limit = spec.default_qr_limit if qr_limit is None else check_finite('qr_limit', qr_limit)
    if qr_limit <= 0:
        raise ValueError(f'qr_limit must be positive, got {qr_limit}.')
    if not isinstance(min_points, int) or min_points < 3:
        raise ValueError(f'min_points must be an integer of at least 3, got {min_points!r}.')
    if kind != 'standard' and qmin is None:
        raise ModelFreeError(
            f'kind={kind!r} needs an explicit qmin above 2*pi/L (L the largest dimension): '
            'below it the transformed intensity rises with Q, so the lower edge of the '
            'cross-section regime cannot be found from the data alone.'
        )

    points = prepare(data, 'Guinier analysis', qmin, qmax)
    positive = points.intensity - background > 0
    q_nonpositive = points.q[~positive]
    points = points.take(positive)
    require_points(points.q.size, min_points, 'Guinier analysis')

    excess = points.intensity - background
    x = points.q**2
    t = np.log(points.q**spec.q_power * excess)
    sigma_t = points.sigma / excess

    if qmax is None:
        n_window = _auto_window(x, t, sigma_t, points.q, spec, qr_limit, min_points)
        window_selected = 'auto'
    else:
        n_window = points.q.size
        window_selected = 'user'
    keep = np.arange(points.q.size) < n_window
    points = points.take(keep)
    x, t, sigma_t, excess = x[keep], t[keep], sigma_t[keep], excess[keep]
    n_nonpositive = int(np.count_nonzero(q_nonpositive <= points.q[-1]))

    line = weighted_polyfit(x, t, sigma_t, 1)
    intercept_log, slope = line.coef
    if slope >= 0:
        raise ModelFreeError(
            'The intensity does not fall with Q in the Guinier window (fitted slope '
            f'{slope:.3g} >= 0). Check the window and the background.'
        )
    radius = math.sqrt(-spec.divisor * slope)
    intercept = math.exp(intercept_log)
    # d(radius, intercept)/d(intercept_log, slope)
    jac = np.array([[0.0, -spec.divisor / (2.0 * radius)], [intercept, 0.0]])
    dof = points.q.size - 2
    cov_line = line.cov * (line.chisq / dof if points.fabricated and dof > 0 else 1.0)
    cov = jac @ cov_line @ jac.T
    sensitivity = np.vstack([jac @ line.hat / excess, np.zeros(points.q.size)])
    background_sensitivity = np.append(jac @ line.hat @ (-1.0 / excess), 1.0)

    result = GuinierResult(
        **points.result_fields(background, n_nonpositive),
        fit=np.empty(0),
        chisq=0.0,
        chisq_linearised=line.chisq,
        cov=cov,
        sensitivity=sensitivity,
        background_sensitivity=background_sensitivity,
        kind=kind,
        radius=radius,
        radius_err=math.sqrt(cov[0, 0]),
        intercept=intercept,
        intercept_err=math.sqrt(cov[1, 1]),
        qr_min=float(points.q[0] * radius),
        qr_max=float(points.q[-1] * radius),
        qr_limit=qr_limit,
        window_selected=window_selected,
    )
    result.fit = result.evaluate(points.q)
    result.chisq = float(np.sum(((points.intensity - result.fit) / points.sigma) ** 2))
    _check(result, spec, x, t, sigma_t)
    return finish(result)


def _auto_window(x, t, sigma_t, q, spec: GuinierKind, qr_limit: float, min_points: int) -> int:
    """Number of points in the chosen window (see :func:`guinier`)."""
    best = None
    smallest_qr = math.inf
    for n in range(min_points, q.size + 1):
        try:
            slope = weighted_polyfit(x[:n], t[:n], sigma_t[:n], 1).coef[1]
        except ModelFreeError:
            continue
        if slope >= 0:
            continue
        qr = q[n - 1] * math.sqrt(-spec.divisor * slope)
        smallest_qr = min(smallest_qr, qr)
        if qr <= qr_limit:
            best = n
        elif best is not None:
            break
    if best is not None:
        return best
    if math.isinf(smallest_qr):
        raise ModelFreeError(
            'The intensity does not fall with Q in any window from the lowest point: '
            'no Guinier region. Check the background and the low-Q edge.'
        )
    raise ModelFreeError(
        f'No window satisfies q*R <= {qr_limit:g}: the smallest value reached is '
        f'{smallest_qr:.3g}. The Guinier region was not measured; lower qmin data '
        'are needed (or pass qmax to force a window and read the qr_limit warning).'
    )


def _check(result: GuinierResult, spec: GuinierKind, x, t, sigma_t) -> None:
    """Append the window and curvature warnings."""
    add = result.warnings.append
    if result.qr_max > result.qr_limit:
        add(
            (
                'qr_limit',
                f'q_max*R = {result.qr_max:.3g} exceeds the limit {result.qr_limit:g}; '
                'the Guinier law is biased in this window.',
            )
        )
    if result.kind == 'standard' and result.qr_min > LOW_Q_COVERAGE_LIMIT:
        add(
            (
                'low_q_coverage',
                f'q_min*Rg = {result.qr_min:.3g} > {LOW_Q_COVERAGE_LIMIT:g}: the data do not '
                'reach the Guinier plateau; a larger structure may be present that these '
                'data cannot show.',
            )
        )
    if result.n_points < FEW_POINTS:
        add(
            (
                'few_points',
                f'Only {result.n_points} points in the Guinier window (fewer than '
                f'{FEW_POINTS}); the curvature check is skipped.',
            )
        )
        return
    if result.kind != 'standard':
        return

    try:
        quad = weighted_polyfit(x, t, sigma_t, 2)
    except ModelFreeError:
        return
    if quad.coef[1] >= 0:  # the quadratic fit has no Guinier slope: c is undefined
        return
    c, c_err = _curvature(quad, result)
    result.curvature, result.curvature_err = c, c_err
    if c < CURVATURE_NEGATIVE and CURVATURE_NEGATIVE - c > CURVATURE_SIGMAS * c_err:
        add(
            (
                'curvature_negative',
                f'Curvature c = {c:.3g} +/- {c_err:.2g}: the intensity turns down at low Q '
                '(interparticle repulsion?). The Rg from this window is biased low.',
            )
        )
    elif c > CURVATURE_POSITIVE and c - CURVATURE_POSITIVE > CURVATURE_SIGMAS * c_err:
        add(
            (
                'curvature_positive',
                f'Curvature c = {c:.3g} +/- {c_err:.2g}: upward curvature, from an elongated '
                'or flexible particle, polydispersity or aggregation. The Rg depends on the '
                'window; compare with qr_limit=1.0.',
            )
        )


def _curvature(quad: LinearFit, result: GuinierResult) -> tuple[float, float]:
    """c = d/(9b²) from ln I = a + b·x + d·x², with its delta-method error."""
    _, b, d = quad.coef
    grad = np.array([0.0, -2.0 * d / (9.0 * b**3), 1.0 / (9.0 * b**2)])
    cov = quad.cov
    if result.uncertainties_fabricated and result.n_points > 3:
        cov = cov * quad.chisq / (result.n_points - 3)
    return float(d / (9.0 * b**2)), float(math.sqrt(grad @ cov @ grad))
