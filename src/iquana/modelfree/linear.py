"""Weighted polynomial least squares with covariance and data sensitivity."""

from dataclasses import dataclass

import numpy as np

from .result import ModelFreeError


@dataclass(slots=True)
class LinearFit:
    """Coefficients in increasing powers of x, with what propagation needs.

    ``cov`` is (XᵀWX)⁻¹ and ``hat`` is d(coef)/dy, shape (k, n), so a change
    δy in the fitted values moves the coefficients by ``hat @ δy``.
    ``chisq`` is the weighted residual sum in the fitted space.
    """

    coef: np.ndarray
    cov: np.ndarray
    hat: np.ndarray
    chisq: float


def weighted_polyfit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, degree: int) -> LinearFit:
    """Fit ``y = sum(c_k x^k)`` with weights 1/sigma², through an SVD.

    Columns are normalised before the pseudo-inverse, so x = q² ~ 1e-4 loses no
    precision. A window where the columns are not independent (all points at
    one Q) raises :class:`ModelFreeError`.
    """
    design = np.vander(x, degree + 1, increasing=True)
    weighted = design / sigma[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    if np.any(norms == 0) or np.linalg.matrix_rank(weighted / norms) < degree + 1:
        raise ModelFreeError(
            'The fit window is degenerate (too few distinct Q values for a '
            f'degree-{degree} fit). Widen the window.'
        )
    solver = np.linalg.pinv(weighted / norms) / norms[:, None]
    hat = solver / sigma[None, :]
    coef = hat @ y
    residuals = (y - design @ coef) / sigma
    return LinearFit(coef=coef, cov=solver @ solver.T, hat=hat, chisq=float(residuals @ residuals))
