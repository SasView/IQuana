"""Result objects and the error type of the model-free analyses.

Two levels: :class:`ModelFreeResult` is what all five analyses return;
:class:`ModelFreeFit` adds what a fit has (parameters, covariance, χ² and a
:meth:`~ModelFreeFit.report`), so the invariant and the Kratky curve, which
are not fits, cannot inherit a fit report by accident.
"""

import math
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, NamedTuple

import numpy as np

from ..data.prepare import FABRICATED_SIGMA_FORMULA
from ..report import FitReport, json_safe


class ModelFreeError(ValueError):
    """The data cannot give a meaningful answer to the requested analysis."""


class GuinierKind(NamedTuple):
    """How one Guinier law is linearised: ln(q^power · I) = ln c - q²R²/divisor."""

    q_power: int
    divisor: float
    radius_name: str
    intercept_name: str
    default_qr_limit: float


GUINIER_KINDS = {
    'standard': GuinierKind(0, 3.0, 'rg', 'i0', 1.3),
    'rod': GuinierKind(1, 2.0, 'rc', 'intercept', 1.0),
    'sheet': GuinierKind(2, 1.0, 'rt', 'intercept', 1.0),
}

# 10⁻⁸ (cm⁻¹ to Å⁻¹) divided by 10⁻¹² (delta_sld² in (10⁻⁶ Å⁻²)² to Å⁻⁴).
ABSOLUTE_SCALE_FACTOR = 1e4


def _num(value: Any) -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.6g}'


def _pm(value: Any, err: Any) -> str:
    if err is None or not math.isfinite(float(err)):
        return _num(value)
    return f'{_num(value)} +/- {float(err):.3g}'


@dataclass(kw_only=True)
class ModelFreeResult:
    """What every model-free analysis returns.

    ``q``, ``intensity`` and ``sigma`` are the points the analysis used, sorted
    by Q; ``index`` gives their positions in the dataset. ``qmin``/``qmax`` are
    the first and last of those points. ``warnings`` holds ``(code, message)``
    pairs; the codes are listed in :mod:`iquana.modelfree`.
    """

    title: ClassVar[str] = 'Model-free result'
    _internal: ClassVar[tuple[str, ...]] = ()

    qmin: float
    qmax: float
    background: float
    data_digest: str
    n_total: int
    index: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    warnings: list[tuple[str, str]] = field(default_factory=list)
    uncertainties_fabricated: bool = False
    n_dropped_points: int = 0
    n_nonpositive_points: int = 0

    @property
    def accepted(self) -> np.ndarray:
        """Boolean mask over the dataset's points: True where the point was used."""
        mask = np.zeros(self.n_total, dtype=bool)
        mask[self.index] = True
        return mask

    @property
    def n_points(self) -> int:
        return int(self.q.size)

    @property
    def warning_codes(self) -> list[str]:
        return [code for code, _ in self.warnings]

    # -- summary -----------------------------------------------------------

    def _summary_rows(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def format_summary(self) -> str:
        """Return an ASCII table of the results, the inputs and every warning."""
        rows = self._summary_rows() + [
            ('Q window (1/Ang)', f'{self.qmin:.6g} to {self.qmax:.6g}'),
            ('Points used', str(self.n_points)),
        ]
        if self.n_dropped_points:
            rows.append(('Points dropped (invalid)', str(self.n_dropped_points)))
        if self.n_nonpositive_points:
            rows.append(('Points dropped (I - B <= 0)', str(self.n_nonpositive_points)))
        width = max(len(label) for label, _ in rows)
        lines = [self.title, '-' * len(self.title)]
        lines.extend(f'{label:<{width}}  {value}' for label, value in rows)
        lines.extend(f'WARNING [{code}]: {message}' for code, message in self.warnings)
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.format_summary()

    def to_dict(self) -> dict[str, Any]:
        """All fields as strictly JSON-serializable data (NaN becomes None)."""
        payload = {'type': type(self).__name__}
        payload.update(
            {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._internal}
        )
        payload['warnings'] = [{'code': c, 'message': m} for c, m in self.warnings]
        return json_safe(payload)

    # -- export ------------------------------------------------------------

    def _csv_columns(self) -> list[tuple[str, np.ndarray]]:
        raise NotImplementedError

    def save_csv(self, filename: str) -> None:
        """Write the summary as a commented header, then one row per point used."""
        columns = self._csv_columns()
        with open(filename, 'w', encoding='utf-8') as f:
            for line in self.format_summary().splitlines():
                f.write(f'# {line}\n')
            f.write('#\n')
            f.write(','.join(name for name, _ in columns) + '\n')
            for row in zip(*(values for _, values in columns), strict=True):
                f.write(','.join(_csv_cell(value) for value in row) + '\n')

    def plot(self, data: Any, show: bool | None = None):
        """Plot the analysis over the dataset it was run on.

        The dataset is passed explicitly, as for ``PrResult.plot_fit``: the
        result keeps only the points it used, and the plot also shows the
        others, dimmed. Same display convention as ``plot_results()``.
        """
        raise NotImplementedError


def _csv_cell(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f'{float(value):.6e}'


@dataclass(kw_only=True)
class ModelFreeFit(ModelFreeResult):
    """A model-free fit: Guinier, power law or Porod.

    ``chisq`` is Σ((I − I_fit)/σ)² in intensity space, as for a model fit;
    ``chisq_linearised`` is what the linearised fit minimised. ``cov`` covers
    the fitted parameters (``cov_labels``). ``sensitivity`` is
    d(parameters)/d(intensity) over the points used, for every parameter in
    :attr:`parameters` order, background last; ``background_sensitivity`` is
    d(parameters)/dB when the background was held fixed.
    """

    _internal: ClassVar[tuple[str, ...]] = ('sensitivity', 'background_sensitivity')
    model_label: ClassVar[str] = 'model-free fit'
    x_label: ClassVar[str] = 'Q (Å⁻¹)'
    y_label: ClassVar[str] = 'I(Q)'
    log_axes: ClassVar[bool] = False

    fit: np.ndarray
    chisq: float
    chisq_linearised: float
    cov: np.ndarray
    sensitivity: np.ndarray
    background_sensitivity: np.ndarray | None = None
    method: str = 'weighted linear least squares'
    cov_source: str = 'linear least squares'
    message: str = 'closed-form solution'

    # -- parameters --------------------------------------------------------

    def _param_entries(self) -> list[tuple[str, float, float | None, bool]]:
        """(name, value, stderr, fitted) for every parameter, background last."""
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        """Parameters in the fit engines' shape, as :class:`FitReport` reads them."""
        from ..fitting.bumps_engine import format_uncertainty

        block = {}
        for name, value, err, fitted in self._param_entries():
            stderr = float(err) if fitted and err is not None else 0.0
            block[name] = {
                'value': float(value),
                'stderr': stderr,
                'formatted': (
                    format_uncertainty(float(value), stderr)
                    if fitted and math.isfinite(stderr) and stderr > 0
                    else f'{float(value):.6g}' + ('' if fitted else ' (fixed)')
                ),
                'fixed': not fitted,
                'linked_to': None,
            }
        return block

    @property
    def theta(self) -> np.ndarray:
        """All parameter values, in :attr:`parameters` order."""
        return np.array([value for _, value, _, _ in self._param_entries()], dtype=float)

    @property
    def cov_labels(self) -> list[str]:
        return [name for name, _, _, fitted in self._param_entries() if fitted]

    @property
    def n_free(self) -> int:
        return len(self.cov_labels)

    @property
    def dof(self) -> int:
        return self.n_points - self.n_free

    @property
    def reduced_chisq(self) -> float:
        return self.chisq / self.dof if self.dof > 0 else float('nan')

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload['parameters'] = json_safe(self.parameters)
        payload['cov_labels'] = self.cov_labels
        payload['reduced_chisq'] = json_safe(self.reduced_chisq)
        return payload

    # -- model -------------------------------------------------------------

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        """The fitted intensity, background included, on any Q grid."""
        raise NotImplementedError

    def _display(self, q, intensity, sigma) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The (x, y, dy) this fit is plotted and exported in."""
        raise NotImplementedError

    def _csv_columns(self) -> list[tuple[str, np.ndarray]]:
        x, y, dy = self._display(self.q, self.intensity, self.sigma)
        _, fit_y, _ = self._display(self.q, self.fit, self.sigma)
        return [
            ('q', self.q),
            ('x', x),
            ('y', y),
            ('dy', dy),
            ('fit', fit_y),
            ('residual', (y - fit_y) / dy),
        ]

    def plot(self, data: Any, show: bool | None = None):
        from ..plotting import plot_modelfree_fit

        return plot_modelfree_fit(data, self, show=show)

    # -- report ------------------------------------------------------------

    def _derived_notes(self) -> list[str]:
        return []

    def _summary_rows(self) -> list[tuple[str, str]]:
        rows = [
            (f'{name} ({"fitted" if fitted else "fixed"})', _pm(value, err if fitted else None))
            for name, value, err, fitted in self._param_entries()
        ]
        rows.extend((note, '') for note in self._derived_notes())
        rows.append(('Reduced chi2', f'{self.reduced_chisq:.4g} ({self.dof} dof)'))
        return rows

    def report(self) -> FitReport:
        """The fit as a :class:`~iquana.report.FitReport`, like a model fit's."""
        weighting = (
            f'fabricated {FABRICATED_SIGMA_FORMULA}' if self.uncertainties_fabricated else 'dI'
        )
        return FitReport(
            model=self.model_label,
            engine='modelfree',
            method=self.method,
            resolution='not applied',
            weighting_note=weighting,
            chisq=self.chisq,
            reduced_chisq=self.reduced_chisq,
            n_points=self.n_points,
            n_free=self.n_free,
            dof=self.dof,
            converged=True,
            message=self.message,
            parameters=self.parameters,
            cov_labels=self.cov_labels,
            cov=np.array(self.cov, copy=True),
            cov_source=self.cov_source,
            notes=[
                f'Q window {self.qmin:.6g} to {self.qmax:.6g} 1/Ang',
                *self._derived_notes(),
                *(f'[{code}] {message}' for code, message in self.warnings),
            ],
        )


@dataclass(kw_only=True)
class GuinierResult(ModelFreeFit):
    """Result of :func:`iquana.modelfree.guinier`.

    ``radius`` is Rg, Rc or Rt and ``intercept`` is I(0) or the cross-section
    intensity, by ``kind``. The kind-specific properties (``rg``, ``i0``,
    ``sphere_radius``, ``cross_section_radius``, ``thickness``) raise
    :class:`ModelFreeError` for the wrong kind.
    """

    x_label: ClassVar[str] = 'Q² (Å⁻²)'

    kind: str
    radius: float
    radius_err: float
    intercept: float
    intercept_err: float
    qr_min: float
    qr_max: float
    qr_limit: float
    window_selected: str
    curvature: float | None = None
    curvature_err: float | None = None

    @property
    def title(self) -> str:  # type: ignore[override]
        return f'Guinier analysis ({self.kind})'

    @property
    def model_label(self) -> str:  # type: ignore[override]
        return 'guinier' if self.kind == 'standard' else f'guinier_{self.kind}'

    @property
    def y_label(self) -> str:  # type: ignore[override]
        return ('ln I', 'ln(Q·I)', 'ln(Q²·I)')[GUINIER_KINDS[self.kind].q_power]

    def _require(self, kind: str, name: str) -> None:
        if self.kind != kind:
            raise ModelFreeError(
                f'{name} is defined for kind={kind!r}; this result is {self.kind!r}.'
            )

    @property
    def rg(self) -> float:
        self._require('standard', 'rg')
        return self.radius

    @property
    def rg_err(self) -> float:
        self._require('standard', 'rg_err')
        return self.radius_err

    @property
    def i0(self) -> float:
        self._require('standard', 'i0')
        return self.intercept

    @property
    def i0_err(self) -> float:
        self._require('standard', 'i0_err')
        return self.intercept_err

    @property
    def sphere_radius(self) -> float:
        """√(5/3)·Rg: the radius of a homogeneous sphere with this Rg."""
        self._require('standard', 'sphere_radius')
        return math.sqrt(5.0 / 3.0) * self.radius

    @property
    def cross_section_radius(self) -> float:
        """√2·Rc: the radius of a solid circular cross-section with this Rc."""
        self._require('rod', 'cross_section_radius')
        return math.sqrt(2.0) * self.radius

    @property
    def thickness(self) -> float:
        """√12·Rt: the thickness of a homogeneous sheet with this Rt."""
        self._require('sheet', 'thickness')
        return math.sqrt(12.0) * self.radius

    def _param_entries(self):
        spec = GUINIER_KINDS[self.kind]
        return [
            (spec.radius_name, self.radius, self.radius_err, True),
            (spec.intercept_name, self.intercept, self.intercept_err, True),
            ('background', self.background, None, False),
        ]

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        spec = GUINIER_KINDS[self.kind]
        decay = np.exp(-((q * self.radius) ** 2) / spec.divisor)
        return self.background + self.intercept * decay / q**spec.q_power

    def _display(self, q, intensity, sigma):
        q = np.asarray(q, dtype=float)
        excess = np.asarray(intensity, dtype=float) - self.background
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log(q ** GUINIER_KINDS[self.kind].q_power * excess)
            dy = np.asarray(sigma, dtype=float) / excess
        return q**2, y, dy

    def _derived_notes(self):
        notes = [f'q*R from {self.qr_min:.3g} to {self.qr_max:.3g} (limit {self.qr_limit:g})']
        if self.kind == 'standard':
            notes.append(f'sphere radius sqrt(5/3)*Rg = {self.sphere_radius:.6g} Ang')
        elif self.kind == 'rod':
            notes.append(f'cross-section radius sqrt(2)*Rc = {self.cross_section_radius:.6g} Ang')
        else:
            notes.append(f'thickness sqrt(12)*Rt = {self.thickness:.6g} Ang')
        if self.curvature is not None:
            notes.append(f'curvature c = {self.curvature:.3g} +/- {self.curvature_err:.2g}')
        return notes


@dataclass(kw_only=True)
class PowerLawResult(ModelFreeFit):
    """Result of :func:`iquana.modelfree.power_law`: I − B = A·q⁻ⁿ."""

    title: ClassVar[str] = 'Power-law analysis'
    model_label: ClassVar[str] = 'power_law'
    y_label: ClassVar[str] = 'I(Q) − B'
    log_axes: ClassVar[bool] = True

    exponent: float
    exponent_err: float
    prefactor: float
    prefactor_err: float
    background_err: float | None = None
    background_fitted: bool = False

    def _param_entries(self):
        return [
            ('prefactor', self.prefactor, self.prefactor_err, True),
            ('exponent', self.exponent, self.exponent_err, True),
            ('background', self.background, self.background_err, self.background_fitted),
        ]

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        return self.prefactor * q ** (-self.exponent) + self.background

    def _display(self, q, intensity, sigma):
        return np.asarray(q), np.asarray(intensity) - self.background, np.asarray(sigma)


@dataclass(kw_only=True)
class PorodResult(ModelFreeFit):
    """Result of :func:`iquana.modelfree.porod`: I = Kp·q⁻⁴ + B.

    ``free_exponent`` comes from an independent three-parameter power-law fit
    over the same window, the check behind the ``porod_exponent`` warning.
    """

    title: ClassVar[str] = 'Porod analysis'
    model_label: ClassVar[str] = 'porod'
    y_label: ClassVar[str] = 'Q⁴·I (Å⁻⁴·cm⁻¹)'

    porod_constant: float
    porod_constant_err: float
    background_err: float
    free_exponent: float | None = None
    free_exponent_err: float | None = None

    def specific_surface(self, delta_sld: float) -> float:
        """S/V = Kp × 10⁴ / (2π·d²) in Å⁻¹, d the SLD difference in 10⁻⁶ Å⁻².

        Needs intensity on absolute scale (cm⁻¹). The relative uncertainty is
        that of ``porod_constant``.
        """
        from .common import check_delta_sld

        d = check_delta_sld(delta_sld)
        return self.porod_constant * ABSOLUTE_SCALE_FACTOR / (2.0 * math.pi * d**2)

    def _param_entries(self):
        return [
            ('porod_constant', self.porod_constant, self.porod_constant_err, True),
            ('background', self.background, self.background_err, True),
        ]

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        return self.porod_constant / q**4 + self.background

    def _display(self, q, intensity, sigma):
        q = np.asarray(q, dtype=float)
        return q, q**4 * np.asarray(intensity), q**4 * np.asarray(sigma)

    def _derived_notes(self):
        if self.free_exponent is None:
            return ['free power-law exponent: not determined']
        return [
            f'free power-law exponent {self.free_exponent:.3f} +/- {self.free_exponent_err:.2g}'
        ]


@dataclass(kw_only=True)
class KratkyCurve(ModelFreeResult):
    """Result of :func:`iquana.modelfree.kratky`: a transform, not a fit.

    ``x, y, dy`` are q, q²·(I − B) or, when ``dimensionless``, q·Rg and
    (q·Rg)²·(I − B)/I(0). ``peak_position``/``peak_height`` locate the data's
    maximum (refined by a parabola through the top three points), or are None
    when the curve has no interior maximum.
    """

    title: ClassVar[str] = 'Kratky curve'

    x: np.ndarray
    y: np.ndarray
    dy: np.ndarray
    dimensionless: bool
    peak_position: float | None = None
    peak_height: float | None = None

    def _summary_rows(self):
        form = 'dimensionless (q*Rg)^2*I/I(0)' if self.dimensionless else 'q^2*(I - B)'
        return [
            ('Form', form),
            ('Background', _num(self.background)),
            ('Peak position', _num(self.peak_position)),
            ('Peak height', _num(self.peak_height)),
        ]

    def _csv_columns(self):
        return [('x', self.x), ('y', self.y), ('dy', self.dy)]

    def plot(self, data: Any = None, show: bool | None = None):
        """Plot the curve. *data* is accepted for symmetry and not needed."""
        from ..plotting import plot_kratky

        return plot_kratky(self, show=show)


@dataclass(kw_only=True)
class InvariantResult(ModelFreeResult):
    """Result of :func:`iquana.modelfree.invariant`: Q* = ∫q²(I − B)dq in three parts.

    Q* is in (intensity unit)·Å⁻³. The part errors are for orientation: the
    parts share data points, so they do not add in quadrature to ``qstar_err``.
    Derived quantities are None when the inputs they need were not given.
    ``*_curve`` arrays are (q, q²·(I − B)) of the extrapolations.
    """

    title: ClassVar[str] = 'Scattering invariant'

    qstar: float
    qstar_err: float
    low_q_part: float
    low_q_part_err: float
    measured_part: float
    measured_part_err: float
    high_q_part: float
    high_q_part_err: float
    low_q_method: str | None = None
    high_q_method: str | None = None
    low_q_curve: np.ndarray | None = None
    high_q_curve: np.ndarray | None = None
    delta_sld: float | None = None
    known_volume_fraction: float | None = None
    phi_product: float | None = None
    phi_product_err: float | None = None
    volume_fraction: float | None = None
    volume_fraction_err: float | None = None
    volume_fraction_complement: float | None = None
    specific_surface_porod: float | None = None
    specific_surface_porod_err: float | None = None
    specific_surface_invariant: float | None = None
    specific_surface_invariant_err: float | None = None
    scale_check: float | None = None

    @property
    def low_q_fraction(self) -> float:
        return self.low_q_part / self.qstar

    @property
    def measured_fraction(self) -> float:
        return self.measured_part / self.qstar

    @property
    def high_q_fraction(self) -> float:
        return self.high_q_part / self.qstar

    @property
    def specific_surface(self) -> float | None:
        """S/V in Å⁻¹ by the scale-free route when available, else the Porod route."""
        if self.specific_surface_invariant is not None:
            return self.specific_surface_invariant
        return self.specific_surface_porod

    def _summary_rows(self):
        def part(label, value, err, method):
            share = f'{100 * value / self.qstar:.1f} %'
            return (label, f'{_pm(value, err)}  ({share}, {method or "not included"})')

        rows = [
            ('Q* (I-unit/Ang^3)', _pm(self.qstar, self.qstar_err)),
            part('  low-Q part', self.low_q_part, self.low_q_part_err, self.low_q_method),
            part('  measured part', self.measured_part, self.measured_part_err, 'trapezoid'),
            part('  high-Q part', self.high_q_part, self.high_q_part_err, self.high_q_method),
            ('Background', _num(self.background)),
        ]
        if self.delta_sld is not None:
            rows += [
                ('delta SLD (1e-6/Ang^2, exact)', _num(self.delta_sld)),
                ('phi(1-phi) (abs. scale, cm^-1)', _pm(self.phi_product, self.phi_product_err)),
                (
                    'Volume fraction (or complement)',
                    f'{_pm(self.volume_fraction, self.volume_fraction_err)} '
                    f'(or {_num(self.volume_fraction_complement)})',
                ),
            ]
        if self.known_volume_fraction is not None:
            rows.append(('Known volume fraction (exact)', _num(self.known_volume_fraction)))
        if self.specific_surface_porod is not None:
            rows.append(
                (
                    'S/V, Porod route (1/Ang)',
                    _pm(self.specific_surface_porod, self.specific_surface_porod_err),
                )
            )
        if self.specific_surface_invariant is not None:
            rows.append(
                (
                    'S/V, scale-free route (1/Ang)',
                    _pm(self.specific_surface_invariant, self.specific_surface_invariant_err),
                )
            )
        if self.scale_check is not None:
            rows.append(('Scale check (measured/expected)', _num(self.scale_check)))
        rows.append(('Note', 'part errors share data and do not add in quadrature'))
        return rows

    def _csv_columns(self):
        q_parts = [self.q]
        values = [self.q**2 * (self.intensity - self.background)]
        sigmas = [self.q**2 * self.sigma]
        parts = [np.full(self.q.size, 'measured')]
        for name, curve in (('low_q', self.low_q_curve), ('high_q', self.high_q_curve)):
            if curve is not None:
                q_parts.append(curve[0])
                values.append(curve[1])
                sigmas.append(np.full(curve.shape[1], np.nan))
                parts.append(np.full(curve.shape[1], name))
        return [
            ('q', np.concatenate(q_parts)),
            ('integrand', np.concatenate(values)),
            ('sigma', np.concatenate(sigmas)),
            ('part', np.concatenate(parts)),
        ]

    def plot(self, data: Any = None, show: bool | None = None):
        """Plot q²·(I − B) with the extrapolated areas shaded. *data* is not needed."""
        from ..plotting import plot_invariant

        return plot_invariant(self, show=show)
