"""SESANS data: the array constructor, and one canonical form on entry.

SESANS (and reduced SEMSANS) data is a sasdata ``Data1D`` with ``isSesans``
set: ``x`` is the spin-echo length, ``y`` the normalised depolarisation
ln(P/P₀)/(tλ²) and ``dy`` its uncertainty. sasmodels turns any I(q) model into
that quantity through a Hankel transform, which reads three more things from
the dataset: the units of ``x``, one wavelength per point
(``source.wavelength``) and the acceptance angle (``sample.zacceptance``).

Every SESANS dataset IQuana holds goes through :func:`prepare_sesans` first, so
these are always in Å, Å and radians, and are present and valid. Two things
make that necessary rather than tidy:

- sasmodels converts other units in the wrong direction (an acceptance in
  degrees is not turned into radians), so only the canonical form is safe;
- sasmodels fails on a single wavelength number, which is what a reactor
  measurement has.

:func:`check_ses_row_order` guards a sasdata reader defect: a ``.ses`` file
whose rows are not in increasing spin-echo length loads with each wavelength
attached to the wrong point.
"""

import copy
import math
import os
from typing import Any

import numpy as np
from sasdata.dataloader.data_info import Data1D

from ..console import ANGSTROM
from .provenance import add_process

#: Factors to Å for the length units a file may state.
_LENGTH_UNITS = {'': 1.0, 'A': 1.0, 'Å': 1.0, 'Ang': 1.0, 'nm': 10.0, 'um': 1e4, 'µm': 1e4}
#: Factors to radians for the angle units a file may state.
_ANGLE_UNITS = {
    '': 1.0,
    'radians': 1.0,
    'radian': 1.0,
    'rad': 1.0,
    'degrees': math.pi / 180,
    'degree': math.pi / 180,
    'deg': math.pi / 180,
}
#: Factors to Å⁻² cm⁻¹ for the depolarisation units, written without spaces.
_DEPOLARISATION_UNITS = {
    '': 1.0,
    'A-2cm-1': 1.0,
    'Å-2cm-1': 1.0,
    'A^-2cm^-1': 1.0,
    'Å⁻²cm⁻¹': 1.0,
    'nm-2cm-1': 0.01,
    'nm^-2cm^-1': 0.01,
}

LENGTH_UNIT = 'A'
ANGLE_UNIT = 'radians'
DEPOLARISATION_UNIT = 'A-2 cm-1'


def _factor(table: dict[str, float], unit: Any, what: str) -> float:
    """The factor to the canonical unit; spaces in *unit* are ignored."""
    text = '' if unit is None else str(unit).strip()
    if text.replace(' ', '') not in table:
        known = ', '.join(sorted(name for name in table if name))
        raise ValueError(f'Unknown unit {text!r} for the {what}. IQuana reads: {known}.')
    return table[text.replace(' ', '')]


def prepare_sesans(data: Any) -> Any:
    """Bring SESANS *data* to Å, radians and Å⁻² cm⁻¹, and check what the transform reads.

    Writes onto *data* (the caller passes its own copy) but replaces its
    ``source`` and ``sample`` with copies first, so the object they came from
    is never changed. A single wavelength becomes one per point.

    Raises:
        ValueError: An unknown unit, a missing or invalid wavelength or
            acceptance, or spin-echo lengths that are not finite, positive and
            strictly increasing. The message names what to fix.
    """
    data.source = copy.copy(data.source)
    data.sample = copy.copy(data.sample)

    length = _factor(_LENGTH_UNITS, getattr(data, '_xunit', ''), 'spin-echo length')
    x = np.asarray(data.x, dtype=float) * length
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError('Spin-echo lengths must be finite and positive.')
    if np.any(np.diff(x) <= 0):
        raise ValueError('Spin-echo lengths must be strictly increasing. Sort the data first.')
    data.x = x
    if data.dx is not None:
        data.dx = np.asarray(data.dx, dtype=float) * length

    depolarisation = _factor(_DEPOLARISATION_UNITS, getattr(data, '_yunit', ''), 'depolarisation')
    data.y = np.asarray(data.y, dtype=float) * depolarisation
    if data.dy is not None:
        data.dy = np.asarray(data.dy, dtype=float) * depolarisation

    data.source.wavelength = _wavelengths(data.source, x.size)
    data.source.wavelength_unit = LENGTH_UNIT
    data.sample.zacceptance = (_acceptance(data.sample), ANGLE_UNIT)

    data.xaxis(getattr(data, '_xaxis', '') or 'Spin-echo length', LENGTH_UNIT)
    data.yaxis(getattr(data, '_yaxis', '') or 'ln(P/P0)/(t lambda^2)', DEPOLARISATION_UNIT)
    return data


def _wavelengths(source: Any, n_points: int) -> np.ndarray:
    values = getattr(source, 'wavelength', None)
    if values is None:
        raise ValueError(
            'SESANS data needs the neutron wavelength (source.wavelength): the transform '
            'uses it. Give it with data_ops.sesans(..., wavelength=...).'
        )
    factor = _factor(_LENGTH_UNITS, getattr(source, 'wavelength_unit', ''), 'wavelength')
    values = np.asarray(values, dtype=float) * factor
    if values.ndim == 0:
        values = np.full(n_points, float(values))
    if values.shape != (n_points,):
        raise ValueError(
            f'SESANS data needs one wavelength, or one per point; got {values.size} '
            f'for {n_points} points.'
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Wavelengths must be finite and positive.')
    return values


def _acceptance(sample: Any) -> float:
    value, unit = (getattr(sample, 'zacceptance', None) or (0, ''))[:2]
    angle = float(value) * _factor(_ANGLE_UNITS, unit, 'acceptance angle')
    if not math.isfinite(angle) or angle <= 0:
        raise ValueError(
            'SESANS data needs the acceptance angle (sample.zacceptance, Theta_zmax in a '
            '.ses file), finite and positive: the largest scattering angle the '
            'measurement collects. See docs/sesans.md for how to work it out.'
        )
    return angle


def wavelength(data: Any) -> np.ndarray:
    """The per-point wavelengths of prepared SESANS *data*, in Å."""
    return np.asarray(data.source.wavelength, dtype=float)


def acceptance(data: Any) -> float:
    """The acceptance angle of prepared SESANS *data*, in radians."""
    return float(data.sample.zacceptance[0])


def describe_transform(data: Any, ascii_only: bool = False) -> str:
    """'wavelength 2.11 Å, acceptance 0.0168 rad': what the transform of *data* uses."""
    values = wavelength(data)
    low, high = float(values.min()), float(values.max())
    text = f'{low:.4g}' if math.isclose(low, high) else f'{low:.4g} to {high:.4g}'
    unit = 'A' if ascii_only else ANGSTROM
    return f'wavelength {text} {unit}, acceptance {acceptance(data):.4g} rad'


def check_ses_row_order(filename: str) -> None:
    """Refuse a ``.ses`` file whose rows are not in increasing spin-echo length.

    sasdata sorts the spin-echo lengths of such a file but leaves the
    wavelengths in file order, so every point would carry another point's
    wavelength, and nothing on the loaded object would show it. A file that is
    not a ``.ses`` text file is left alone.
    """
    try:
        with open(filename, encoding='utf-8', errors='replace') as handle:
            lines = handle.read().splitlines()
    except OSError:
        return
    starts = [i for i, line in enumerate(lines) if line.strip().upper() == 'BEGIN_DATA']
    if not starts:
        return
    lengths = []
    for line in lines[starts[0] + 2 :]:  # skip the column header
        fields = line.split()
        if not fields:
            continue
        try:
            lengths.append(float(fields[0]))
        except ValueError:
            break
    if np.any(np.diff(lengths) <= 0):
        raise ValueError(
            f'{os.path.basename(filename)}: the data rows are not in increasing spin-echo '
            'length. sasdata sorts the lengths but not the wavelengths that belong to '
            'them, so the loaded data would pair points with the wrong wavelength. Sort '
            'the rows of the file by SpinEchoLength first.'
        )


def build_sesans(
    x: Any,
    y: Any,
    dy: Any,
    wavelength: Any,
    acceptance: float,
    dx: Any = None,
) -> Data1D:
    """A SESANS ``Data1D`` in the canonical units, not yet checked.

    The common ground of :func:`sesans` and of an analysis file that embeds
    one; both pass the result through ``normalize_sans_data``.
    """
    data = Data1D(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        dx=None if dx is None else np.asarray(dx, dtype=float),
        dy=None if dy is None else np.asarray(dy, dtype=float),
        isSesans=True,
    )
    data.xaxis('Spin-echo length', LENGTH_UNIT)
    data.yaxis('ln(P/P0)/(t lambda^2)', DEPOLARISATION_UNIT)
    data.source.wavelength = np.asarray(wavelength, dtype=float)
    data.source.wavelength_unit = LENGTH_UNIT
    data.sample.zacceptance = (float(acceptance), ANGLE_UNIT)
    data.sample.yacceptance = (float(acceptance), ANGLE_UNIT)
    return data


def sesans(
    spin_echo_length: Any,
    *,
    polarisation: Any = None,
    depolarisation: Any = None,
    error: Any = None,
    wavelength: Any,
    acceptance: float,
    thickness: float | None = None,
) -> Data1D:
    """Build a fit-ready SESANS dataset from arrays, such as reduced SEMSANS data.

    Give the measurement in one of two forms:

    - ``polarisation=``: the ratio P/P₀ (or Vs/V₀ for SEMSANS). It is turned
      into the normalised depolarisation ln(P/P₀)/(tλ²) that sasmodels fits,
      which needs the sample *thickness*. ``error`` is the one-standard-
      deviation uncertainty of the ratio, propagated to first order:
      σ = σ_P / (P·tλ²). Errors are treated as independent between points; a
      shared reference measurement correlates them.
    - ``depolarisation=``: already ln(P/P₀)/(tλ²) in Å⁻² cm⁻¹ (the
      ``Depolarisation`` column of a ``.ses`` file). ``error`` is its
      uncertainty and *thickness* is not used, so giving it is an error.

    A ratio above one is accepted: it occurs in measured data. The British
    spelling of the two keywords follows the ``.ses`` format.

    Args:
        spin_echo_length: Spin-echo lengths in Å, strictly increasing.
        polarisation: P/P₀ per point, all positive.
        depolarisation: ln(P/P₀)/(tλ²) per point, in Å⁻² cm⁻¹.
        error: One standard deviation of whichever of the two is given. Without
            it the bumps engine and ``fit_bayesian()`` refuse to fit; the scipy
            engine fits unweighted with a warning.
        wavelength: Neutron wavelength in Å: one number, or one per point
            (time of flight).
        acceptance: The largest scattering angle the measurement collects, in
            radians. See ``docs/sesans.md``.
        thickness: Sample thickness in cm. Required with ``polarisation=``.

    Returns:
        A ``Data1D`` with ``isSesans`` set, ready for ``IQuana.set_data()``.

    Raises:
        ValueError: For a missing or doubled measurement, arrays of different
            lengths, a ratio at or below zero, or an invalid wavelength,
            acceptance or thickness.
    """
    from .loader import normalize_sans_data

    if (polarisation is None) == (depolarisation is None):
        raise ValueError('Give exactly one of polarisation= and depolarisation=.')
    x = _column(spin_echo_length, 'spin_echo_length')
    n_points = x.size
    given = _column(
        polarisation if depolarisation is None else depolarisation,
        'polarisation' if depolarisation is None else 'depolarisation',
        n_points,
    )
    sigma = None if error is None else _column(error, 'error', n_points)
    lam = np.asarray(wavelength, dtype=float)
    if lam.ndim == 0:
        lam = np.full(n_points, float(lam))
    lam = _column(lam, 'wavelength', n_points)

    if depolarisation is not None:
        if thickness is not None:
            raise ValueError(
                'thickness= is used only to normalise polarisation=; the depolarisation is '
                'already divided by it. Leave it out.'
            )
        y, dy, what = given, sigma, 'depolarisation'
    else:
        t = float(thickness) if thickness is not None else math.nan
        if not math.isfinite(t) or t <= 0:
            raise ValueError('polarisation= needs the sample thickness in cm, finite and positive.')
        bad = int(np.sum(~(given > 0)))
        if bad:
            raise ValueError(
                f'{bad} of {n_points} polarisation ratios are zero, negative or not finite; '
                'the logarithm needs ratios above zero.'
            )
        scale = t * lam**2
        y = np.log(given) / scale
        dy = None if sigma is None else sigma / (given * scale)
        what = f'polarisation, thickness {t:g} cm'

    data = build_sesans(x, y, dy, lam, acceptance)
    data.title = 'SESANS data'
    if thickness is not None:
        data.sample.thickness, data.sample.thickness_unit = float(thickness), 'cm'
    add_process(data, f'sesans({what}, {n_points} points)')
    return normalize_sans_data(data)


def _column(values: Any, name: str, n_points: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f'{name} must be a non-empty 1D array.')
    if n_points is not None and array.size != n_points:
        raise ValueError(f'{name} has {array.size} values for {n_points} spin-echo lengths.')
    return array


def with_finer_transform(data: Any) -> Any:
    """A copy of SESANS *data* with one extra point a quarter into the first step.

    sasmodels takes the largest q of its transform from the first step between
    spin-echo lengths, so the extra point raises it fourfold. The point has no
    measurement (NaN, masked), so it is never fitted or compared; it only
    changes how finely the transform is computed. Used by the sensitivity
    check after a fit.
    """
    finer = copy.deepcopy(data)
    x = np.asarray(data.x, dtype=float)
    finer.x = np.insert(x, 1, x[0] + (x[1] - x[0]) / 4)
    finer.y = np.insert(np.asarray(data.y, dtype=float), 1, np.nan)
    for name in ('dy', 'dx'):
        values = getattr(data, name, None)
        if values is not None:
            setattr(finer, name, np.insert(np.asarray(values, dtype=float), 1, 1.0))
    mask = getattr(data, 'mask', None)
    if mask is not None:
        finer.mask = np.insert(np.asarray(mask, dtype=bool), 1, True)
    lam = wavelength(data)
    finer.source.wavelength = np.insert(lam, 1, lam[0])
    return finer
