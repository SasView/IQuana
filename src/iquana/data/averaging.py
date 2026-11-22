"""2D to 1D: circular, sector and annular averages of a detector image.

Three averages, each available as a function in :mod:`iquana.data_ops` (for a
bare dataset) and as a method of :class:`~iquana.IQuana` (which adds the
fitter's pixel selection, Q range and resolution setting):

- :func:`circular_average` gives I(Q) over the whole detector;
- :func:`sector_average` gives I(Q) inside a sector, by default together with
  the sector on the other side of the beam;
- :func:`annular_average` gives I(φ) inside a ring, as an
  :class:`AnnularAverage`, whose :meth:`~AnnularAverage.fit_sin2` measures the
  size and direction of a twofold anisotropy.

The rules, in short (``docs/usage.md``, "From 2D to 1D", has the full text):

- **Angles** are in degrees, anticlockwise from +Qx.
- **Pixels** are not split: a pixel belongs to the bin its centre falls in.
  Each bin holds the plain mean of its pixels, with σ = √(Σσᵢ²)/N.
- **One grid per image.** The bin edges depend on the image and on ``bins``
  only, never on the sector, the mask or the Q range, so every cut of one
  image has the same Q values and can be subtracted from another at once.
  ``bins=<a cut or any Data1D>`` puts a cut of another file on that grid.
- **NaN for empty bins.** A bin without pixels stays in the result as NaN,
  which IQuana masks everywhere. So does a bin the mask or a Q limit cuts
  through (an *edge bin*): its fixed Q value no longer describes its pixels.

Warning codes, stored on the result as ``(code, message)`` and emitted with
the code in brackets, as the model-free analyses do:

``uncertainties_from_scatter``, ``zero_uncertainty_pixels``, ``one_sided``,
``sides_differ``, ``partial_ring``, ``uneven_ring``, ``weak_anisotropy`` and,
from dataset arithmetic, ``shared_pixels``.
"""

import copy
import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sasdata.dataloader.data_info import Data1D

from ..console import DEGREE, INVERSE_ANGSTROM, OK, PHI, PLUS_MINUS, SIGMA, logger
from .loader import has_real_data, normalize_sans_data
from .provenance import add_process
from .selection import ORIGIN_Q, Selection, coordinate_digest, mask_from, validate_2d
from .view import intensity, is_2d, q_magnitude, uncertainty

__all__ = [
    'AnisotropyFit',
    'AnnularAverage',
    'annular_average',
    'circular_average',
    'sector_average',
]

#: A bin whose pixels have a mean |Q| further than this from the bin's Q value
#: is an edge bin and is dropped. 1 % would also drop fully covered bins of a
#: regular detector (1.1 % from the pixel lattice alone).
EDGE_BIN_TOLERANCE = 0.02
#: Without dI, a bin needs this many pixels for its scatter to be an error bar.
MIN_SCATTER_PIXELS = 5
#: ``uneven_ring``: largest spread of the per-angle mean |Q|, peak to peak.
UNEVEN_RING_LIMIT = 0.05
#: ``sides_differ``: both a median relative difference above this ...
SIDES_RELATIVE_LIMIT = 0.02
#: ... and a χ² this improbable for identical sides, over at least ...
SIDES_P_VALUE = 1e-3
#: ... this many bins where both sides have data.
SIDES_MIN_BINS = 10
#: ``weak_anisotropy``: B / σ_B below this leaves the direction undetermined.
WEAK_ANISOTROPY = 3.0
#: Directions on the detector that ``phi`` accepts by name.
DIRECTIONS = {'horizontal': 0.0, 'vertical': 90.0}

ANNULAR_NOT_A_CURVE = (
    'An annular average is intensity against angle and cannot be fitted with a '
    'scattering model. Use sector_average() for I(Q), or ring.fit_sin2().'
)

Mask = np.ndarray | Callable[[np.ndarray, np.ndarray], Any] | None


# =============================================================================
# The image: which pixels may be used, and their values
# =============================================================================


@dataclass(slots=True)
class Image:
    """The per-pixel values of a 2D dataset that an average reads."""

    data: Any
    usable: np.ndarray
    q: np.ndarray
    phi: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None
    dq: np.ndarray | None
    resolution: str

    @property
    def label(self) -> str:
        for attr in ('filename', 'title'):
            value = getattr(self.data, attr, None)
            if value:
                return os.path.basename(str(value))
        return 'image'


def image_for(data: Any, excluded: np.ndarray, resolution: Any = None) -> Image:
    """Collect what an average needs from *data*.

    Args:
        data: A 2D dataset.
        excluded: Pixels not to use, True = excluded.
        resolution: The fitter's ``ResolutionSetting``, or None to use the
            file's own radial width (``dqx_data``) when it has one.
    """
    q = q_magnitude(data)
    values = intensity(data)
    sigma = uncertainty(data)
    dq, described = _pixel_resolution(data, q, resolution)
    phi = np.mod(np.degrees(np.arctan2(data.qy_data, data.qx_data)), 360.0)
    return Image(
        data=data,
        usable=~np.asarray(excluded, dtype=bool) & np.isfinite(values) & (q > ORIGIN_Q),
        q=q,
        phi=phi,
        intensity=values,
        sigma=sigma if has_real_data(sigma) else None,
        dq=dq,
        resolution=described,
    )


def _pixel_resolution(data: Any, q: np.ndarray, setting: Any) -> tuple[np.ndarray | None, str]:
    """The radial 1-sigma width of every pixel, and where it came from."""
    mode = 'data' if setting is None else setting.mode
    if mode == 'pinhole':
        return setting.dq_over_q * q, f'pinhole {SIGMA}_q/q = {setting.dq_over_q:g}'
    if mode == 'none':
        return None, "none (set_resolution('none'))"
    column = getattr(data, 'dqx_data', None)
    if has_real_data(column):
        return np.abs(np.asarray(column, dtype=float)), 'from the file'
    return None, "not in the file, use set_resolution('pinhole', dq_over_q=...)"


def excluded_pixels(data: Any, mask: Mask, mask_convention: str | None, caller: str) -> np.ndarray:
    """The exclusion mask of a bare dataset: its own mask plus *mask*."""
    _require_image(data, caller)
    validate_2d(data)
    excluded = Selection.from_data(data, mask_convention).excluded()
    if mask is not None:
        excluded = excluded | mask_from(data, mask)
    return excluded


def _require_image(data: Any, caller: str) -> None:
    if not is_2d(data):
        raise TypeError(
            f'{caller}() needs a 2D detector image (qx_data, qy_data, data); '
            f'got {type(data).__name__} without them.'
        )


# =============================================================================
# The Q grid
# =============================================================================


@dataclass(frozen=True, slots=True)
class QGrid:
    """Bin edges, half-open ``[a, b)``, and the Q value of each bin."""

    edges: np.ndarray
    q: np.ndarray
    default: bool = False

    @property
    def size(self) -> int:
        return int(self.q.size)

    def index(self, q: np.ndarray) -> np.ndarray:
        """The bin of each value, or -1 outside the grid."""
        index = np.searchsorted(self.edges, q, side='right') - 1
        index[(index < 0) | (index >= self.size)] = -1
        return index


def ring_q(edges: np.ndarray) -> np.ndarray:
    """Mean |Q| of a uniformly covered ring between consecutive edges."""
    a, b = edges[:-1], edges[1:]
    return (2.0 / 3.0) * (b**3 - a**3) / (b**2 - a**2)


_DEFAULT_EDGES: dict[str, np.ndarray] = {}
_CACHE_SIZE = 8


def q_grid(data: Any, bins: Any = None, log: bool = False) -> QGrid:
    """The grid for *bins* (see :func:`circular_average`) on the image *data*."""
    if bins is None:
        if log:
            raise ValueError('log=True needs a number of bins, for example bins=50.')
        edges = default_edges(data)
        return QGrid(edges, ring_q(edges), default=True)
    if getattr(bins, 'x', None) is not None:
        if log:
            raise ValueError('log=True cannot be combined with bins=<a curve>.')
        return _curve_grid(bins)
    if isinstance(bins, bool):
        raise TypeError('bins must be None, a number of bins, an array of edges or a curve.')
    if isinstance(bins, (int, np.integer)):
        return _equal_grid(data, int(bins), log)
    if log:
        raise ValueError('log=True cannot be combined with explicit bin edges.')
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('Bin edges must be a 1D array of at least two values.')
    if not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('Bin edges must be finite, not negative and strictly increasing.')
    return QGrid(edges, ring_q(edges))


def _equal_grid(data: Any, n: int, log: bool) -> QGrid:
    if n < 1:
        raise ValueError(f'bins must be at least 1, got {n}.')
    q = q_magnitude(data)
    q = q[q > ORIGIN_Q]
    low, high = float(q.min()), float(q.max())
    edges = np.geomspace(low, high, n + 1) if log else np.linspace(low, high, n + 1)
    edges[-1] = np.nextafter(high, np.inf)  # the outermost pixel is inside
    return QGrid(edges, ring_q(edges))


def _curve_grid(curve: Any) -> QGrid:
    """The grid of an earlier cut, or one around the Q values of any curve."""
    x = np.asarray(curve.x, dtype=float)
    averaging = getattr(curve, 'averaging', None)
    if isinstance(averaging, dict) and np.size(averaging.get('edges')) == x.size + 1:
        return QGrid(np.asarray(averaging['edges'], dtype=float), x.copy())
    if x.size < 2 or not np.all(np.isfinite(x)) or x[0] <= 0 or np.any(np.diff(x) <= 0):
        raise ValueError(
            'A curve used as bins= needs at least two finite, positive and strictly '
            'increasing Q values.'
        )
    middle = (x[1:] + x[:-1]) / 2
    first = max(x[0] - (middle[0] - x[0]), 0.0)
    last = x[-1] + (x[-1] - middle[-1])
    return QGrid(np.concatenate([[first], middle, [last]]), x.copy())


def default_edges(data: Any) -> np.ndarray:
    """Edges from 0 outward, each bin as wide as the local pixel pitch.

    The pitch of a pixel is the distance to its nearest neighbour. Each bin is
    as wide as the median pitch of the pixels it holds, rounded to three
    figures, so a merged image with two pixel sizes gets narrow bins where
    its pixels are small and wide ones where they are large. Kept per image,
    because the neighbour search takes about a second for a megapixel.
    """
    key = coordinate_digest(data)
    edges = _DEFAULT_EDGES.get(key)
    if edges is None:
        edges = _pitch_edges(np.asarray(data.qx_data, float), np.asarray(data.qy_data, float))
        if len(_DEFAULT_EDGES) >= _CACHE_SIZE:
            _DEFAULT_EDGES.pop(next(iter(_DEFAULT_EDGES)))
        _DEFAULT_EDGES[key] = edges
    return edges


def _pitch_edges(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    points = np.column_stack([qx, qy])
    pitch = cKDTree(points).query(points, k=2)[0][:, 1]
    q = np.hypot(qx, qy)
    order = np.argsort(q)
    q, pitch = q[order], pitch[order]
    pitch[pitch <= 0] = np.nan  # duplicated coordinates
    # The width of a regular image, for an image with no usable pitch at all.
    fallback = math.sqrt(np.ptp(qx) * np.ptp(qy) / q.size) or q[-1] or 1.0

    edges = [0.0]
    while edges[-1] <= q[-1]:
        start = edges[-1]
        width = pitch[min(np.searchsorted(q, start), q.size - 1)]
        width = width if np.isfinite(width) else fallback
        for _ in range(3):  # settle on the median pitch of the bin's own pixels
            low, high = np.searchsorted(q, [start, start + width])
            inside = pitch[low:high]
            if np.isfinite(inside).any():
                width = float(np.nanmedian(inside))
        edges.append(start + float(f'{width:.3g}'))
    return np.array(edges)


def _pitches(widths: np.ndarray) -> list[float]:
    """The distinct pixel pitches among bin widths, merging those within 25 %."""
    groups: list[list[float]] = []
    for width in np.unique(np.round(widths, 12)):
        if groups and width <= 1.25 * groups[-1][-1]:
            groups[-1].append(width)
        else:
            groups.append([width])
    return [float(np.median(group)) for group in groups]


# =============================================================================
# Binning
# =============================================================================


@dataclass(slots=True)
class _Binned:
    n: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    q_mean: np.ndarray
    dq: np.ndarray | None = None
    q_spread: np.ndarray | None = None


def _bin(image: Image, picked: np.ndarray, index: np.ndarray, size: int, q_ref=None) -> _Binned:
    """Mean intensity, its uncertainty and mean |Q| per bin, NaN where empty.

    Without dI the uncertainty is the standard error of the pixels' mean, and
    a bin with fewer than :data:`MIN_SCATTER_PIXELS` pixels is NaN. With
    *q_ref* (the Q value of each bin) the result also carries the spread of
    |Q| about it and, when the pixels have widths, the bin's dQ.
    """
    index = index[picked]
    values = image.intensity[picked]
    q = image.q[picked]
    n = np.bincount(index, minlength=size)[:size]

    def mean(weights: np.ndarray) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.bincount(index, weights, minlength=size)[:size] / n

    average = mean(values)
    with np.errstate(invalid='ignore', divide='ignore'):
        if image.sigma is None:
            spread = mean((values - average[index]) ** 2) * n / (n - 1)
            sigma = np.sqrt(spread / n)
            few = n < MIN_SCATTER_PIXELS
            average[few] = np.nan
            sigma[few] = np.nan
        else:
            pixel_sigma = image.sigma[picked]
            variance = np.where(np.isfinite(pixel_sigma), pixel_sigma**2, 0.0)
            sigma = np.sqrt(mean(variance) * n) / n
    binned = _Binned(n=n, intensity=average, sigma=sigma, q_mean=mean(q))
    if q_ref is not None:
        offset = mean((q - q_ref[index]) ** 2)
        binned.q_spread = np.sqrt(offset)
        if image.dq is not None:
            binned.dq = np.sqrt(mean(image.dq[picked] ** 2) + offset)
    return binned


# =============================================================================
# I(Q): circular and sector averages
# =============================================================================


def circular_average(
    data: Any,
    qmin: float | None = None,
    qmax: float | None = None,
    bins: Any = None,
    log: bool = False,
    mask: Mask = None,
    mask_convention: str | None = None,
) -> Data1D:
    """Average a 2D image over the whole detector: I(Q).

    Args:
        data: A 2D dataset, as ``set_data()`` accepts it.
        qmin, qmax: Use only pixels with qmin ≤ |Q| ≤ qmax. A bin that a limit
            cuts through is left out whole (NaN).
        bins: The Q grid. ``None`` (default) makes bins as wide as the local
            pixel pitch, from Q = 0 outward. An integer makes that many equal
            bins from the smallest |Q| to the largest (equal in log Q with
            ``log=True``). An array gives the edges. A curve reuses its grid:
            an earlier cut (its edges and Q values), or any ``Data1D`` (edges
            halfway between its Q values, and its Q values as x).
        log: Logarithmic bins; needs an integer *bins*.
        mask: More pixels to exclude: a boolean array (True = excluded) or
            ``f(qx, qy)``, as ``IQuana.set_mask()`` takes. Adds to the
            dataset's own mask.
        mask_convention: How to read the dataset's own mask, as in
            ``set_data()``.

    Returns:
        A fit-ready ``Data1D``. Every bin of the grid is a row; a bin without
        data is NaN and masked. ``averaging`` holds the details (edges, pixel
        counts, mean |Q| per bin, warnings).

    Raises:
        TypeError: For 1D data.
        ValueError: For a bad argument, or when no usable pixel is left.
    """
    excluded = excluded_pixels(data, mask, mask_convention, 'circular_average')
    return radial_average(image_for(data, excluded), qmin=qmin, qmax=qmax, bins=bins, log=log)


def sector_average(
    data: Any,
    phi: float | str,
    half_width: float = 10.0,
    mirror: bool = True,
    qmin: float | None = None,
    qmax: float | None = None,
    bins: Any = None,
    log: bool = False,
    mask: Mask = None,
    mask_convention: str | None = None,
) -> Data1D:
    """Average a 2D image inside a sector: I(Q) along one direction.

    The sector holds the pixels with phi − half_width ≤ angle < phi +
    half_width, angles in degrees anticlockwise from +Qx. On data with
    I = A + B·sin²θ (θ from the field) a sector across the field returns
    A + B·½(1 + sin 2w / 2w) and one along it A + B·½(1 − sin 2w / 2w), for
    half-width w in radians:

    ==========  =============  ============
    half_width  across: A + B  along: A + B
    ==========  =============  ============
    5°          × 0.997        × 0.003
    10°         × 0.990        × 0.010
    20°         × 0.960        × 0.040
    45°         × 0.818        × 0.182
    ==========  =============  ============

    At low Q a narrow sector holds few pixels (a pixel is in or out by its
    centre), so the opening actually sampled there is not exactly the one
    asked for: use coarser ``bins`` or a wider sector when the low-Q points
    matter. ``averaging['n_pixels']`` gives the count per bin.

    Args:
        phi: Centre of the sector in degrees, any finite value (370 means 10),
            or ``'horizontal'`` (0) or ``'vertical'`` (90). These are
            directions on the detector; which one is the field depends on the
            instrument.
        half_width: Half the opening, in degrees: above 0, at most 90 with
            ``mirror=True`` and at most 180 without.
        mirror: Also use the sector on the other side of the beam (phi +
            180), pooled with this one. The result never has negative Q.
        Other arguments: as :func:`circular_average`.

    Returns:
        A fit-ready ``Data1D``, as :func:`circular_average`.
    """
    excluded = excluded_pixels(data, mask, mask_convention, 'sector_average')
    return radial_average(
        image_for(data, excluded),
        sector=(phi, half_width, mirror),
        qmin=qmin,
        qmax=qmax,
        bins=bins,
        log=log,
    )


def radial_average(
    image: Image,
    *,
    sector: tuple[Any, float, bool] | None = None,
    qmin: float | None = None,
    qmax: float | None = None,
    bins: Any = None,
    log: bool = False,
) -> Data1D:
    """The circular (``sector=None``) or sector average of an :class:`Image`."""
    if sector is not None:
        sector = _check_sector(*sector)
    qmin, qmax = _check_limits(qmin, qmax, required=False)
    grid = q_grid(image.data, bins, log)
    index = grid.index(image.q)

    stages = [('masked or unusable', image.usable)]
    base = image.usable & _within(image.q, qmin, qmax)
    stages.append(('outside the Q range', base))
    base &= index >= 0
    stages.append(('outside the bins', base))
    side_a, side_b = base, np.zeros_like(base)
    if sector is not None:
        centre, half_width, mirror = sector
        side_a = base & _in_sector(image.phi, centre, half_width)
        if mirror:
            side_b = base & _in_sector(image.phi, centre + 180.0, half_width) & ~side_a
    picked = side_a | side_b
    stages.append(('outside the sector', picked))
    if not picked.any():
        raise ValueError(_no_pixels_message(image, stages, sector is not None))

    binned = _bin(image, picked, index, grid.size, grid.q)
    keep = binned.n > 0
    if qmin is not None:
        keep &= grid.edges[:-1] >= qmin
    if qmax is not None:
        keep &= grid.edges[1:] <= qmax
    with np.errstate(invalid='ignore', divide='ignore'):
        edge = keep & (np.abs(binned.q_mean / grid.q - 1) > EDGE_BIN_TOLERANCE)
    keep &= ~edge
    with_data = keep & np.isfinite(binned.intensity)
    if not with_data.any():
        raise ValueError(_no_bins_message(image, binned, edge))

    found: list[tuple[str, str]] = []
    _check_pixels(image, picked, found)
    if sector is not None and sector[2]:
        _check_sides(image, side_a, side_b, index, grid.size, found)

    for column in (binned.intensity, binned.sigma, binned.dq):
        if column is not None:
            column[~with_data] = np.nan
    used = grid.edges[:-1][with_data].min(), grid.edges[1:][with_data].max()
    averaging = {
        'kind': 'circular' if sector is None else 'sector',
        'phi': None if sector is None else sector[0],
        'half_width': None if sector is None else sector[1],
        'mirror': None if sector is None else sector[2],
        'qmin': float(used[0]),
        'qmax': float(used[1]),
        'edges': grid.edges.copy(),
        'n_pixels': binned.n,
        'q_mean': binned.q_mean,
        'q_spread': binned.q_spread,
        'edge_bins': np.flatnonzero(edge),
        'n_used': int(picked.sum()),
        'n_total': int(picked.size),
        'resolution': image.resolution,
        'warnings': found,
        'image': coordinate_digest(image.data),
        'pixels': np.packbits(picked),
    }
    curve = _curve(image, grid, binned, with_data, averaging, _title(image, sector))
    logger.info(_radial_log_line(image, grid, averaging, with_data, sector, (qmin, qmax)))
    _emit(found)
    return curve


def _check_sector(phi: Any, half_width: Any, mirror: Any) -> tuple[float, float, bool]:
    if isinstance(phi, str):
        try:
            phi = DIRECTIONS[phi.lower()]
        except KeyError:
            raise ValueError(
                f"phi must be a number of degrees, 'horizontal' or 'vertical'; got {phi!r}."
            ) from None
    if isinstance(phi, bool) or not isinstance(phi, (int, float, np.number)):
        raise TypeError(f'phi must be a number of degrees, got {phi!r}.')
    if not math.isfinite(phi):
        raise ValueError(f'phi must be finite, got {phi}.')
    half_width = float(half_width)
    limit = 90.0 if mirror else 180.0
    if not 0.0 < half_width <= limit:
        raise ValueError(
            f'half_width must be above 0 and at most {limit:g} degrees with '
            f'mirror={bool(mirror)}, got {half_width:g}.'
        )
    return float(phi) % 360.0, half_width, bool(mirror)


def _check_limits(qmin: Any, qmax: Any, required: bool) -> tuple[float | None, float | None]:
    limits = []
    for name, value in (('qmin', qmin), ('qmax', qmax)):
        if value is None:
            if required:
                raise ValueError(f'{name} is required.')
            limits.append(None)
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and not negative, got {value:g}.')
        limits.append(value)
    if limits[0] is not None and limits[1] is not None and limits[0] >= limits[1]:
        raise ValueError(f'qmin ({limits[0]:g}) must be smaller than qmax ({limits[1]:g}).')
    return limits[0], limits[1]


def _within(q: np.ndarray, qmin: float | None, qmax: float | None) -> np.ndarray:
    inside = np.ones(q.size, dtype=bool)
    if qmin is not None:
        inside &= q >= qmin
    if qmax is not None:
        inside &= q <= qmax
    return inside


def _in_sector(phi: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    """Pixels with centre − half_width ≤ angle < centre + half_width, across 0°/360°."""
    return np.mod(phi - (centre - half_width), 360.0) < 2.0 * half_width


def _no_pixels_message(image: Image, stages: list[tuple[str, np.ndarray]], sector: bool) -> str:
    counts = ', '.join(f'{int(left.sum())} left after removing those {why}' for why, left in stages)
    message = f'No usable pixel in the region: {image.usable.size} pixels, {counts}.'
    if sector:
        message += ' Angles are in degrees, anticlockwise from +Qx.'
    return message


def _no_bins_message(image: Image, binned: _Binned, edge: np.ndarray) -> str:
    if image.sigma is None and not np.any(binned.n >= MIN_SCATTER_PIXELS):
        return (
            f'The image has no dI, and no bin holds the {MIN_SCATTER_PIXELS} pixels needed '
            'to estimate an uncertainty from their scatter. Use coarser bins= or a wider '
            'sector.'
        )
    return (
        f'Every bin with pixels was dropped ({int(edge.sum())} edge bins, cut through by '
        'the mask or a Q limit). Use coarser bins= or wider Q limits.'
    )


def _title(image: Image, sector: tuple[float, float, bool] | None) -> str:
    if sector is None:
        return f'{image.label} circular average'
    centre, half_width, mirror = sector
    return f'{image.label} sector {centre:g}°±{half_width:g}°' + ('' if mirror else ' one side')


def _curve(
    image: Image,
    grid: QGrid,
    binned: _Binned,
    with_data: np.ndarray,
    averaging: dict[str, Any],
    title: str,
) -> Data1D:
    """The fit-ready Data1D of a cut, with the image's metadata and provenance."""
    curve = Data1D(x=grid.q.copy(), y=binned.intensity, dy=binned.sigma, dx=binned.dq)
    source = image.data
    for attr in ('run', 'run_name', 'instrument', 'sample', 'source', 'detector', 'collimation'):
        if hasattr(source, attr):
            setattr(curve, attr, copy.deepcopy(getattr(source, attr)))
    curve.xaxis('\\rm{Q}', 'A^{-1}')
    curve.yaxis('\\rm{Intensity}', getattr(source, '_zunit', '') or '')
    curve.process = copy.deepcopy(list(getattr(source, 'process', None) or []))
    curve.title = curve.filename = title
    # The Q range of the bins with data, so a data summary does not start
    # inside the beam stop.
    curve.qmin = float(grid.q[with_data].min())
    curve.qmax = float(grid.q[with_data].max())
    normalize_sans_data(curve)
    add_process(curve, _description(title, averaging, grid, with_data))
    curve.averaging = averaging
    return curve


def _description(title: str, averaging: dict[str, Any], grid: QGrid, with_data: np.ndarray) -> str:
    """Provenance text, in ASCII so a saved analysis stores it on any system."""
    text = f'{averaging["kind"].capitalize()} average: {title}'.replace('°', ' deg')
    text = text.replace('±', ' +/- ')
    return (
        f'{text}; {averaging["n_used"]} of {averaging["n_total"]} pixels, '
        f'{grid.size} bins ({int(with_data.sum())} with data), '
        f'Q {averaging["qmin"]:.6g} to {averaging["qmax"]:.6g} 1/Ang, '
        f'dQ {averaging["resolution"]}'
    )


def _radial_log_line(
    image: Image,
    grid: QGrid,
    averaging: dict[str, Any],
    with_data: np.ndarray,
    sector: tuple[float, float, bool] | None,
    limits: tuple[float | None, float | None],
) -> str:
    if sector is None:
        what = 'Circular average'
    else:
        centre, half_width, mirror = sector
        what = (
            f'Sector average {PHI} = {centre:g}{DEGREE} {PLUS_MINUS} {half_width:g}{DEGREE}'
            f' ({"mirrored" if mirror else "one side"})'
        )
    n_edge = int(averaging['edge_bins'].size)
    counts = f'{int(with_data.sum())} with data'
    if n_edge:
        counts += f', {n_edge} edge bin{"s" if n_edge > 1 else ""} dropped'
    lines = [
        f'{OK} {what}: {averaging["n_used"]} of {averaging["n_total"]} pixels, '
        f'{grid.size} bins ({counts}), dQ: {image.resolution}'
    ]
    if limits != (None, None):
        lines.append(
            f'  Q {averaging["qmin"]:.6g} to {averaging["qmax"]:.6g} {INVERSE_ANGSTROM} '
            '(whole bins inside the Q limits)'
        )
    if grid.default:
        pitches = _pitches(np.diff(grid.edges))
        if len(pitches) > 1:
            listed = ' and '.join(f'{p:.3g}' for p in pitches)
            lines.append(f'  {len(pitches)} pixel pitches: {listed} {INVERSE_ANGSTROM}')
    if sector is not None and sector[2] and sector[1] == 90.0:
        lines.append('  This is the full circle: circular_average() gives the same.')
    return '\n'.join(lines)


# =============================================================================
# Checks that become warnings
# =============================================================================


def _check_pixels(image: Image, picked: np.ndarray, found: list[tuple[str, str]]) -> None:
    """``uncertainties_from_scatter`` and ``zero_uncertainty_pixels``."""
    if image.sigma is None:
        found.append(
            (
                'uncertainties_from_scatter',
                'The image has no dI: each uncertainty is the standard error of the '
                "mean of the bin's pixels. That scatter also holds the slope of I(Q) "
                'across the bin and any variation with angle, so it overestimates the '
                f'error. Bins with fewer than {MIN_SCATTER_PIXELS} pixels are left out.',
            )
        )
        return
    sigma = image.sigma[picked]
    n_zero = int(np.sum(~np.isfinite(sigma) | (sigma == 0)))
    if n_zero:
        found.append(
            (
                'zero_uncertainty_pixels',
                f'{n_zero} pixels with dI = 0 or not finite are included. Each counts in '
                'the mean but adds nothing to the uncertainty, so the dI of the bins '
                'holding them is too small. If they are dead pixels, exclude them: '
                'fitter.mask_invalid(), or mask= here.',
            )
        )


def _check_sides(
    image: Image,
    side_a: np.ndarray,
    side_b: np.ndarray,
    index: np.ndarray,
    size: int,
    found: list[tuple[str, str]],
) -> None:
    """``one_sided`` and ``sides_differ`` for a mirrored sector."""
    if not side_a.any() or not side_b.any():
        found.append(
            (
                'one_sided',
                'mirror=True, but one of the two opposite sectors holds no usable pixel: '
                'the average uses one side only.',
            )
        )
        return
    a = _bin(image, side_a, index, size)
    b = _bin(image, side_b, index, size)
    variance = a.sigma**2 + b.sigma**2
    common = np.isfinite(a.intensity) & np.isfinite(b.intensity) & (variance > 0)
    n = int(common.sum())
    if n < SIDES_MIN_BINS:
        return
    from scipy.stats import chi2

    difference = a.intensity[common] - b.intensity[common]
    chisq = float(np.sum(difference**2 / variance[common]))
    scale = (np.abs(a.intensity[common]) + np.abs(b.intensity[common])) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = float(np.nanmedian(np.abs(difference) / scale))
    if chi2.sf(chisq, n) < SIDES_P_VALUE and relative > SIDES_RELATIVE_LIMIT:
        found.append(
            (
                'sides_differ',
                f'The two opposite sectors differ by {100 * relative:.1f} % (median) over '
                f'{n} bins, reduced chi2 {chisq / n:.3g}. Possible causes: a beam centre '
                'that is off, gravity at long wavelength, detector efficiency, or '
                'scattering that is not centrosymmetric. Compare the two sides with '
                'mirror=False at phi and at phi + 180.',
            )
        )


def _emit(found: list[tuple[str, str]], stacklevel: int = 4) -> None:
    """Warn at the caller of the public function or fitter method."""
    for code, message in found:
        warnings.warn(f'{message} [{code}]', UserWarning, stacklevel=stacklevel)


def shared_pixels(a: Any, b: Any) -> bool:
    """True when *a* and *b* are cuts of one image that use some of the same pixels."""
    first, second = getattr(a, 'averaging', None), getattr(b, 'averaging', None)
    if not (isinstance(first, dict) and isinstance(second, dict)):
        return False
    if first.get('image') != second.get('image'):
        return False
    pixels_a, pixels_b = first.get('pixels'), second.get('pixels')
    if pixels_a is None or pixels_b is None or np.size(pixels_a) != np.size(pixels_b):
        return False
    return bool(np.any(np.bitwise_and(pixels_a, pixels_b)))


# =============================================================================
# I(φ): the annular average
# =============================================================================


def annular_average(
    data: Any,
    qmin: float,
    qmax: float,
    bins: int = 36,
    mask: Mask = None,
    mask_convention: str | None = None,
) -> 'AnnularAverage':
    """Average a 2D image inside the ring qmin ≤ |Q| ≤ qmax: I(φ).

    Args:
        data: A 2D dataset, as ``set_data()`` accepts it.
        qmin, qmax: The ring, in 1/Å.
        bins: Number of angular bins over the full circle (36 = 10° bins). A
            bin is centred on 0°.
        mask, mask_convention: As :func:`circular_average`.

    Returns:
        An :class:`AnnularAverage`. It is not a ``Data1D``: I(φ) cannot be
        fitted with a scattering model, but :meth:`AnnularAverage.fit_sin2`
        measures its anisotropy.
    """
    excluded = excluded_pixels(data, mask, mask_convention, 'annular_average')
    return angular_average(image_for(data, excluded), qmin, qmax, bins)


def angular_average(image: Image, qmin: float, qmax: float, bins: int = 36) -> 'AnnularAverage':
    """The annular average of an :class:`Image`."""
    qmin, qmax = _check_limits(qmin, qmax, required=True)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f'bins must be a positive number of angular bins, got {bins!r}.')
    bins = int(bins)
    width = 360.0 / bins
    stages = [('masked or unusable', image.usable)]
    picked = image.usable & _within(image.q, qmin, qmax)
    stages.append(('outside the ring', picked))
    if not picked.any():
        raise ValueError(_no_pixels_message(image, stages, sector=False))
    index = (np.floor((image.phi + width / 2) / width).astype(int)) % bins
    binned = _bin(image, picked, index, bins)

    found: list[tuple[str, str]] = []
    _check_pixels(image, picked, found)
    populated = binned.n > 0
    if populated.mean() < 0.5:
        found.append(
            (
                'partial_ring',
                f'Only {int(populated.sum())} of {bins} angular bins hold pixels: less than '
                'half of the ring is on the detector or unmasked.',
            )
        )
    q_mean = binned.q_mean[populated]
    unevenness = float(np.ptp(q_mean) / np.mean(q_mean))
    if unevenness > UNEVEN_RING_LIMIT:
        found.append(
            (
                'uneven_ring',
                f'The mean |Q| differs by {100 * unevenness:.1f} % between angular bins, so '
                'a slope of I(Q) shows as variation with angle. The ring probably reaches '
                'past the detector edge or is narrow compared with the pixels; use a ring '
                'inside the detector, a wider ring, or fewer angular bins.',
            )
        )
    ring = AnnularAverage(
        phi=np.arange(bins) * width,
        intensity=binned.intensity,
        uncertainty=binned.sigma,
        n_pixels=binned.n,
        q_mean=binned.q_mean,
        qmin=qmin,
        qmax=qmax,
        bin_width=width,
        source=image.label,
        n_used=int(picked.sum()),
        n_total=int(picked.size),
        uncertainties_from_scatter=image.sigma is None,
        image=coordinate_digest(image.data),
        pixels=np.packbits(picked),
        warnings=found,
    )
    with_data = int(np.isfinite(ring.intensity).sum())
    logger.info(
        f'{OK} Annular average {qmin:g} to {qmax:g} {INVERSE_ANGSTROM}: {ring.n_used} of '
        f'{ring.n_total} pixels, {bins} angular bins of {width:g}{DEGREE} ({with_data} with data)'
    )
    _emit(found)
    return ring


@dataclass(kw_only=True)
class AnnularAverage:
    """I(φ) inside a ring of a 2D image, from :func:`annular_average`.

    ``phi`` holds the bin centres in degrees (0 to 360, a bin centred on 0),
    ``intensity`` and ``uncertainty`` the mean and its error per bin (NaN
    where a bin has no data), ``q_mean`` the mean |Q| of each bin's pixels.

    Deliberately not a ``Data1D``: ``IQuana.set_data()`` and the model-free
    analyses refuse it, since the angles are not Q values.
    """

    phi: np.ndarray
    intensity: np.ndarray
    uncertainty: np.ndarray
    n_pixels: np.ndarray
    q_mean: np.ndarray
    qmin: float
    qmax: float
    bin_width: float
    source: str
    n_used: int
    n_total: int
    uncertainties_from_scatter: bool = False
    image: str = ''
    pixels: np.ndarray | None = field(default=None, repr=False)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f'{self.source} ring {self.qmin:g} to {self.qmax:g} 1/Ang'

    @property
    def warning_codes(self) -> list[str]:
        return [code for code, _ in self.warnings]

    def format_summary(self) -> str:
        """An ASCII table of the ring, its bins and every warning."""
        with_data = np.isfinite(self.intensity)
        rows = [
            ('Ring (1/Ang)', f'{self.qmin:.6g} to {self.qmax:.6g}'),
            (
                'Angular bins',
                f'{self.phi.size} of {self.bin_width:g} deg, {int(with_data.sum())} with data',
            ),
            ('Pixels used', f'{self.n_used} of {self.n_total}'),
            ('Uncertainties', 'scatter of the pixels' if self.uncertainties_from_scatter else 'dI'),
        ]
        if with_data.any():
            rows.append(
                (
                    'Intensity',
                    f'{np.nanmin(self.intensity):.4g} to {np.nanmax(self.intensity):.4g}',
                )
            )
        return _table(f'Annular average: {self.source}', rows, self.warnings)

    def __str__(self) -> str:
        return self.format_summary()

    def save_csv(self, filename: str) -> None:
        """Write the summary as a commented header, then one row per angular bin."""
        columns = {
            'phi_deg': self.phi,
            'I': self.intensity,
            'dI': self.uncertainty,
            'n_pixels': self.n_pixels,
            'q_mean': self.q_mean,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            for line in self.format_summary().splitlines():
                f.write(f'# {line}\n')
            f.write(','.join(columns) + '\n')
            for row in zip(*columns.values(), strict=True):
                f.write(','.join(f'{float(value):.6e}' for value in row) + '\n')

    def plot(self, show: bool | None = None):
        """I(φ) with error bars, as a plotly figure."""
        from ..plotting2d import plot_annular

        return plot_annular(self, show=show)

    def fit_sin2(self, phi0: float | None = None) -> 'AnisotropyFit':
        """Fit I(φ) = A + B·sin²(φ − φ₀).

        The law of a saturated magnetic sample with the field in the detector
        plane, and the lowest harmonic of any twofold pattern. A weighted
        linear fit of 1, cos 2φ and sin 2φ, corrected for the width of the
        angular bins (exact for bins covered evenly in angle).

        Args:
            phi0: None to fit the direction: B is then positive and φ₀, in
                [0, 180), is **the direction of least intensity**, which is the
                field direction only where the physics makes B positive. Give
                a direction in degrees to keep it fixed; B then keeps its sign.
                For SANSPOL, fit φ₀ on the unpolarised or I⁺ ring and pass it
                to the I⁻ fit, which returns B < 0.

        Raises:
            ValueError: With fewer angular bins with data than parameters + 1.
        """
        fit = _fit_sin2(self, phi0)
        _emit(fit.warnings, stacklevel=3)
        return fit


# =============================================================================
# fit_sin2
# =============================================================================


@dataclass(kw_only=True)
class AnisotropyFit:
    """Result of :meth:`AnnularAverage.fit_sin2`: I(φ) = A + B·sin²(φ − φ₀).

    ``phi0`` is in degrees. With ``phi0_fitted`` it is the direction of least
    intensity and ``b`` is positive; its error is NaN when the anisotropy is
    too weak to fix a direction (warning ``weak_anisotropy``).
    """

    a: float
    a_err: float
    b: float
    b_err: float
    phi0: float
    phi0_err: float
    phi0_fitted: bool
    chisq: float
    n_points: int
    cov: np.ndarray
    cov_labels: list[str]
    ring: AnnularAverage = field(repr=False)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """B / A: the size of the anisotropy relative to the isotropic part."""
        return self.b / self.a if self.a else float('nan')

    @property
    def dof(self) -> int:
        return self.n_points - len(self.cov_labels)

    @property
    def reduced_chisq(self) -> float:
        return self.chisq / self.dof if self.dof > 0 else float('nan')

    @property
    def warning_codes(self) -> list[str]:
        return [code for code, _ in self.warnings]

    def evaluate(self, phi: np.ndarray) -> np.ndarray:
        """A + B·sin²(φ − φ₀) at the angles *phi*, in degrees."""
        return self.a + self.b * np.sin(np.radians(np.asarray(phi, float) - self.phi0)) ** 2

    def _entries(self) -> list[tuple[str, float, float | None, bool]]:
        return [
            ('a', self.a, self.a_err, True),
            ('b', self.b, self.b_err, True),
            ('phi0', self.phi0, self.phi0_err, self.phi0_fitted),
        ]

    def format_summary(self) -> str:
        """An ASCII table of A, B, φ₀ and the fit quality, then every warning."""
        rows = [
            (f'{name} ({"fitted" if fitted else "fixed"})', _pm(value, err if fitted else None))
            for name, value, err, fitted in self._entries()
        ]
        rows += [
            ('B / A', f'{self.ratio:.4g}'),
            ('Reduced chi2', f'{self.reduced_chisq:.4g} ({self.dof} dof)'),
            ('Ring (1/Ang)', f'{self.ring.qmin:.6g} to {self.ring.qmax:.6g}'),
        ]
        if self.phi0_fitted:
            rows.append(('phi0 means', 'the direction of least intensity'))
        return _table('Anisotropy: I = A + B sin^2(phi - phi0)', rows, self.warnings)

    def __str__(self) -> str:
        return self.format_summary()

    def report(self):
        """The fit as a :class:`~iquana.report.FitReport`, like a model-free fit's."""
        from ..report import FitReport, parameter_block

        return FitReport(
            model='sin2',
            engine='analytic',
            method='weighted linear least squares',
            resolution='not applied',
            weighting_note=(
                'scatter of the pixels' if self.ring.uncertainties_from_scatter else 'dI'
            ),
            chisq=self.chisq,
            reduced_chisq=self.reduced_chisq,
            n_points=self.n_points,
            n_free=len(self.cov_labels),
            dof=self.dof,
            converged=True,
            message='closed-form solution',
            parameters=parameter_block(self._entries()),
            cov_labels=list(self.cov_labels),
            cov=np.array(self.cov, copy=True),
            cov_source='linear least squares',
            notes=[
                f'{self.ring.title}, {self.ring.phi.size} angular bins',
                *(f'[{code}] {message}' for code, message in self.warnings),
            ],
        )

    def plot(self, show: bool | None = None):
        """The ring with the fitted curve, as a plotly figure."""
        from ..plotting2d import plot_annular

        return plot_annular(self.ring, fit=self, show=show)


def _fit_sin2(ring: AnnularAverage, phi0: float | None) -> AnisotropyFit:
    ok = np.isfinite(ring.intensity) & np.isfinite(ring.uncertainty) & (ring.uncertainty > 0)
    n_params = 3 if phi0 is None else 2
    if ok.sum() <= n_params:
        raise ValueError(
            f'fit_sin2() needs more than {n_params} angular bins with data and an '
            f'uncertainty; this ring has {int(ok.sum())}.'
        )
    angle = np.radians(ring.phi[ok])
    half = np.radians(ring.bin_width) / 2
    # Averaging cos 2φ over a bin of width w scales it by sin(w)/w.
    damping = math.sin(2 * half) / (2 * half)
    if phi0 is None:
        design = np.column_stack(
            [np.ones_like(angle), damping * np.cos(2 * angle), damping * np.sin(2 * angle)]
        )
    else:
        phi0 = float(phi0) % 180.0
        shifted = 2 * (angle - math.radians(phi0))
        design = np.column_stack([np.ones_like(angle), (1 - damping * np.cos(shifted)) / 2])
    sigma = ring.uncertainty[ok]
    weighted = design / sigma[:, None]
    target = ring.intensity[ok] / sigma
    try:
        cov = np.linalg.inv(weighted.T @ weighted)
    except np.linalg.LinAlgError:
        raise ValueError('fit_sin2(): the angular bins with data do not fix the fit.') from None
    coef = cov @ weighted.T @ target
    chisq = float(np.sum((target - weighted @ coef) ** 2))

    if phi0 is not None:
        a, b = coef
        errors = np.sqrt(np.diag(cov))
        return AnisotropyFit(
            a=float(a),
            a_err=float(errors[0]),
            b=float(b),
            b_err=float(errors[1]),
            phi0=phi0,
            phi0_err=0.0,
            phi0_fitted=False,
            chisq=chisq,
            n_points=int(ok.sum()),
            cov=cov,
            cov_labels=['a', 'b'],
            ring=ring,
        )

    c0, c1, c2 = coef
    r = math.hypot(c1, c2)
    b = 2 * r
    direction = (math.degrees(0.5 * math.atan2(-c2, -c1)) % 180.0) if r > 0 else 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        jacobian = np.array(
            [
                [1.0, -c1 / r, -c2 / r],
                [0.0, 2 * c1 / r, 2 * c2 / r],
                [0.0, -0.5 * c2 / r**2 * 180 / math.pi, 0.5 * c1 / r**2 * 180 / math.pi],
            ]
        )
    cov = jacobian @ cov @ jacobian.T
    errors = np.sqrt(np.abs(np.diag(cov)))
    found: list[tuple[str, str]] = []
    phi0_err = float(errors[2])
    if not (b > WEAK_ANISOTROPY * errors[1]):
        phi0_err = float('nan')
        found.append(
            (
                'weak_anisotropy',
                f'B = {b:.3g} is less than {WEAK_ANISOTROPY:g} times its error '
                f'({errors[1]:.3g}): the direction phi0 is not determined, and B, forced '
                'positive, is biased upward.',
            )
        )
    return AnisotropyFit(
        a=float(c0 - r),
        a_err=float(errors[0]),
        b=float(b),
        b_err=float(errors[1]),
        phi0=float(direction),
        phi0_err=phi0_err,
        phi0_fitted=True,
        chisq=chisq,
        n_points=int(ok.sum()),
        cov=cov,
        cov_labels=['a', 'b', 'phi0'],
        ring=ring,
        warnings=found,
    )


def _pm(value: float, err: float | None) -> str:
    if err is None or not math.isfinite(err):
        return f'{value:.6g}'
    return f'{value:.6g} +/- {err:.3g}'


def _table(title: str, rows: list[tuple[str, str]], found: list[tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    lines = [title, '-' * len(title)]
    lines += [f'{label:<{width}}  {value}' for label, value in rows]
    lines += [f'WARNING [{code}]: {message}' for code, message in found]
    return '\n'.join(lines)
