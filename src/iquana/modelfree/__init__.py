"""
Model-free first-look analysis: Guinier, power law, Porod, Kratky and the invariant.

Plain functions on a 1D dataset (``fitter.data`` or a ``data_ops`` result),
with no sasmodels kernel involved, as for P(r) inversion in
:mod:`iquana.inversion`. Each returns a result object with
``format_summary()``, ``to_dict()``, ``save_csv()`` and ``plot(data)``; the
three fits (Guinier, power law, Porod) also give a ``report()`` like a model
fit's, and their numbers are plain floats ready for ``fitter.set_param()``.

Example:
    >>> from iquana import data_ops, modelfree
    >>> data = data_ops.load('protein.csv')
    >>> p = modelfree.porod(data, qmin=0.15)
    >>> g = modelfree.guinier(data, background=p.background)
    >>> print(g.format_summary())
    >>> k = modelfree.kratky(data, guinier=g)
    >>> inv = modelfree.invariant(data, low_q=g, high_q=p, delta_sld=2.5)

Units and conventions:
    - Q in 1/Å, lengths in Å. ``delta_sld`` is a number in 10⁻⁶ Å⁻², like
      every SLD in IQuana and sasmodels. Absolute quantities (volume fraction,
      Porod-route specific surface) assume intensity in cm⁻¹.
    - An explicit ``qmin``/``qmax`` replaces the dataset's own limit (set by
      ``fitter.set_q_range()``) on that side; the mask is always honoured.
      Points are sorted by Q.
    - ``background`` is a flat level subtracted before the calculation. Chained
      calls must use one background: :func:`kratky` and :func:`invariant`
      inherit it from their input results and refuse a different one.
    - Missing dI: the P(r) rule fabricates sigma, flags the result, and scales
      fit uncertainties by the reduced chi-squared.

Warnings are stored on the result as ``(code, message)`` pairs and also
emitted through :mod:`warnings`. The codes: ``qr_limit``, ``low_q_coverage``,
``curvature_negative``, ``curvature_positive``, ``few_points``,
``short_q_range``, ``low_signal``, ``porod_exponent``, ``extrapolation_low``,
``extrapolation_high``, ``window_not_at_edge``,
``background_uncertainty_excluded``, ``truncated``,
``uncertainties_fabricated``, ``dq_ignored``, ``outside_dataset_range``.

Limitations:
    - Slit-smeared data (dxl/dxw) is refused by every function; pinhole dQ is
      ignored, with a ``dq_ignored`` warning.
    - 2D and SESANS data are refused.
    - Q is assumed to be in 1/Å; nm⁻¹ data gives lengths off by 10.
"""

from ..data.prepare import InsufficientDataError
from .guinier import guinier
from .invariant import invariant
from .kratky import kratky
from .porod import porod
from .power_law import power_law
from .result import (
    GuinierResult,
    InvariantResult,
    KratkyCurve,
    ModelFreeError,
    ModelFreeFit,
    ModelFreeResult,
    PorodResult,
    PowerLawResult,
)

__all__ = [
    'guinier',
    'power_law',
    'porod',
    'kratky',
    'invariant',
    'ModelFreeResult',
    'ModelFreeFit',
    'GuinierResult',
    'PowerLawResult',
    'PorodResult',
    'InvariantResult',
    'KratkyCurve',
    'ModelFreeError',
    'InsufficientDataError',
]
