"""2D to 1D: circular, sector and annular averages (roadmap item 3.3).

The oracles are analytic: constant and sin² images, quadrant images, and the
pixels themselves. sasdata's averagers are not used as an oracle; see
MD/AVERAGING_SASDATA_ISSUES.md for why.
"""

import contextlib
import io
import time
import warnings

import numpy as np
import pytest
from sasdata.dataloader.data_info import Data1D, Data2D

from iquana import (
    AnisotropyFit,
    AnnularAverage,
    IQuana,
    MultiFitter,
    data_ops,
    examples,
    modelfree,
    pr_inversion,
    set_verbosity,
)
from iquana.data.averaging import EDGE_BIN_TOLERANCE, MIN_SCATTER_PIXELS, ring_q


@pytest.fixture(autouse=True)
def quiet():
    set_verbosity('quiet')
    yield
    set_verbosity('info')


def image(intensity=None, n=64, qmax=0.1, err=0.01, dq=None, keep=None, relative=False):
    """A sasdata Data2D on an n x n grid of Q (no pixel at the origin for even n)."""
    axis = np.linspace(-qmax, qmax, n)
    qx, qy = (grid.ravel() for grid in np.meshgrid(axis, axis))
    if keep is not None:
        inside = keep(qx, qy)
        qx, qy = qx[inside], qy[inside]
    values = np.ones(qx.size) if intensity is None else intensity(qx, qy) * np.ones(qx.size)
    if err is None:
        sigma = np.zeros(qx.size)
    else:
        sigma = err * values if relative else np.full(qx.size, float(err))
    data = Data2D(
        data=values,
        err_data=sigma,
        qx_data=qx,
        qy_data=qy,
        q_data=np.hypot(qx, qy),
        mask=np.ones(qx.size, dtype=bool),
    )
    if dq is not None:
        data.dqx_data = dq * np.hypot(qx, qy)
        data.dqy_data = dq * np.hypot(qx, qy)
    data.filename = 'test_image'
    return data


def sin2(a=2.0, b=1.0, phi0=0.0):
    """I = a + b·sin²(φ − φ0), φ the angle of the pixel."""

    def intensity(qx, qy):
        return a + b * np.sin(np.arctan2(qy, qx) - np.radians(phi0)) ** 2

    return intensity


def porod(qx, qy):
    return 1e-4 * np.hypot(qx, qy) ** -4


def fitter_for(data):
    fitter = IQuana()
    fitter.set_data(data)
    return fitter


def finite(curve):
    return np.isfinite(curve.y)


def quietly(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call(*args, **kwargs)


# =============================================================================
# Values
# =============================================================================


class TestValues:
    def test_constant_image_gives_the_constant_and_sigma_over_root_n(self):
        data = image(lambda qx, qy: 3.0, err=0.2)
        for cut in (
            data_ops.circular_average(data),
            data_ops.sector_average(data, 30, half_width=20),
        ):
            ok = finite(cut)
            np.testing.assert_allclose(cut.y[ok], 3.0)
            n = cut.averaging['n_pixels'][ok]
            np.testing.assert_allclose(cut.dy[ok], 0.2 / np.sqrt(n))
        ring = data_ops.annular_average(data, 0.03, 0.06)
        np.testing.assert_allclose(ring.intensity, 3.0)
        np.testing.assert_allclose(ring.uncertainty, 0.2 / np.sqrt(ring.n_pixels))

    @pytest.mark.parametrize('half_width', [5, 10, 20, 45])
    def test_sectors_of_a_sin2_image_follow_the_mixing_table(self, half_width):
        a, b = 2.0, 1.0
        data = image(sin2(a, b), n=256)
        w = np.radians(half_width)
        across = data_ops.sector_average(data, 90, half_width=half_width)
        along = data_ops.sector_average(data, 0, half_width=half_width)
        # Inside the inscribed circle, where the opening is fully on the detector.
        many = (across.averaging['n_pixels'] >= 400) & (across.averaging['edges'][1:] < 0.1)
        np.testing.assert_allclose(
            across.y[many & finite(across)], a + b * 0.5 * (1 + np.sin(2 * w) / (2 * w)), rtol=5e-3
        )
        np.testing.assert_allclose(
            along.y[many & finite(along)], a + b * 0.5 * (1 - np.sin(2 * w) / (2 * w)), rtol=5e-3
        )

    @pytest.mark.parametrize('phi0', [0.0, 35.0, 120.0])
    def test_fit_sin2_returns_a_b_and_the_direction(self, phi0):
        ring = data_ops.annular_average(image(sin2(2.0, 1.0, phi0), n=256), 0.04, 0.08)
        fit = ring.fit_sin2()
        assert isinstance(fit, AnisotropyFit)
        assert fit.a == pytest.approx(2.0, rel=5e-3)
        assert fit.b == pytest.approx(1.0, rel=1e-2)
        assert (fit.phi0 - phi0 + 90) % 180 - 90 == pytest.approx(0.0, abs=0.5)
        assert fit.ratio == pytest.approx(0.5, rel=2e-2)
        assert fit.warnings == []

    def test_fixed_phi0_keeps_the_sign_of_b(self):
        ring = data_ops.annular_average(image(sin2(3.0, -1.0, 0.0), n=128), 0.04, 0.08)
        fit = ring.fit_sin2(phi0=0.0)
        assert not fit.phi0_fitted
        assert fit.b == pytest.approx(-1.0, rel=2e-2)
        free = ring.fit_sin2()
        assert free.b > 0
        assert (free.phi0 - 90 + 90) % 180 - 90 == pytest.approx(0.0, abs=1.0)

    def test_quadrant_image(self):
        bright = image(lambda qx, qy: np.where((qx > 0) & (np.abs(qy) < qx * 0.3), 10.0, 1.0))
        ring = data_ops.annular_average(bright, 0.03, 0.07)
        assert ring.phi[np.nanargmax(ring.intensity)] == 0.0
        towards = data_ops.sector_average(bright, 0, half_width=10, mirror=False)
        away = data_ops.sector_average(bright, 180, half_width=10, mirror=False)
        assert np.nanmin(towards.y) == pytest.approx(10.0)
        assert np.nanmax(away.y) == pytest.approx(1.0)

    def test_sectors_that_tile_the_circle_add_up_to_the_circular_average(self):
        data = image(porod, err=1.0)
        circle = data_ops.circular_average(data)
        sectors = [
            data_ops.sector_average(data, centre, half_width=30, mirror=False)
            for centre in range(0, 360, 60)
        ]
        counts = sum(s.averaging['n_pixels'] for s in sectors)
        np.testing.assert_array_equal(counts, circle.averaging['n_pixels'])
        assert sum(s.averaging['n_used'] for s in sectors) == circle.averaging['n_used']
        # Pixel-weighted mean of the sector means, where the sectors are not dropped.
        total = sum(np.nan_to_num(s.y) * s.averaging['n_pixels'] for s in sectors)
        whole = np.logical_and.reduce(
            [np.isfinite(s.y) | (s.averaging['n_pixels'] == 0) for s in sectors]
        )
        ok = finite(circle) & whole
        np.testing.assert_allclose((total / counts)[ok], circle.y[ok], rtol=1e-12)

    def test_bin_q_matches_the_pixels(self):
        data = image(n=128)
        for cut in (data_ops.circular_average(data), data_ops.sector_average(data, 90)):
            ok = finite(cut)
            deviation = np.abs(cut.averaging['q_mean'][ok] / cut.x[ok] - 1)
            assert np.median(deviation) < 0.01
            assert deviation.max() <= EDGE_BIN_TOLERANCE

    def test_bin_q_is_the_ring_mean(self):
        edges = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(ring_q(edges), [2 / 3, (2 / 3) * 7 / 3])

    def test_noise_free_sphere_matches_the_1d_model(self):
        data = examples.simulate_2d('sphere', qmax=0.2, npixels=128, noise=0, radius=40)
        cut = quietly(data_ops.circular_average, data, bins=40)
        ok = finite(cut)
        curve = examples.simulate('sphere', q=cut.x[ok], noise=0, radius=40)
        # A bin's mean intensity differs from the curve at its Q by the
        # curvature across the bin; small away from the minima.
        ratio = cut.y[ok] / curve.y
        assert np.median(np.abs(ratio - 1)) < 0.02


# =============================================================================
# The grid
# =============================================================================


class TestGrid:
    def test_default_grid_on_a_regular_image_is_the_pixel_pitch(self):
        data = image(n=101, qmax=0.1)
        cut = data_ops.circular_average(data)
        widths = np.diff(cut.averaging['edges'])
        np.testing.assert_allclose(widths, 0.002, rtol=1e-9)

    def test_default_grid_follows_both_pitches_of_the_merged_s50_image(self):
        data = examples.load('sanspol_plus')
        cut = quietly(data_ops.circular_average, data)
        edges = cut.averaging['edges']
        widths = np.diff(edges)
        assert np.median(widths[edges[:-1] < 0.09]) == pytest.approx(0.00104, rel=0.02)
        assert np.median(widths[edges[:-1] > 0.12]) == pytest.approx(0.0038, rel=0.05)

    def test_every_cut_of_one_image_has_the_same_grid(self):
        data = image(porod)
        cuts = [
            data_ops.circular_average(data),
            data_ops.sector_average(data, 0),
            data_ops.sector_average(data, 90, mask=lambda qx, qy: np.hypot(qx, qy) < 0.02),
            data_ops.circular_average(data, qmin=0.03, qmax=0.06),
        ]
        for cut in cuts[1:]:
            np.testing.assert_array_equal(cut.x, cuts[0].x)

    def test_integer_and_log_and_explicit_bins(self):
        data = image()
        q = np.hypot(data.qx_data, data.qy_data)
        linear = data_ops.circular_average(data, bins=20)
        assert linear.x.size == 20
        np.testing.assert_allclose(np.diff(linear.averaging['edges']), np.ptp(q) / 20)
        logarithmic = quietly(data_ops.circular_average, data, bins=20, log=True)
        ratios = np.diff(np.log(logarithmic.averaging['edges']))
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)
        edges = np.array([0.01, 0.02, 0.05, 0.08])
        explicit = data_ops.circular_average(data, bins=edges)
        np.testing.assert_array_equal(explicit.averaging['edges'], edges)

    @pytest.mark.parametrize('bins', [None, 7, 'log'])
    def test_the_pixel_at_the_largest_q_is_inside_the_grid(self, bins):
        data = image()
        kwargs = {'bins': 7, 'log': True} if bins == 'log' else {'bins': bins}
        cut = quietly(data_ops.circular_average, data, **kwargs)
        q = np.hypot(data.qx_data, data.qy_data)
        assert cut.averaging['n_used'] == q.size
        assert cut.averaging['edges'][-1] > q.max()

    def test_a_beam_stop_through_a_bin_makes_it_an_edge_bin(self):
        data = examples.load('p123_2d')
        stop = 0.0105  # inside the bin [0.010, 0.012)
        cut = quietly(data_ops.circular_average, data, mask=lambda qx, qy: np.hypot(qx, qy) < stop)
        edges = cut.averaging['edges']
        cut_through = np.flatnonzero((edges[:-1] < stop) & (edges[1:] > stop))
        assert set(cut_through) <= set(cut.averaging['edge_bins'])
        assert np.all(np.isnan(cut.y[cut_through]))
        assert np.all(cut.mask[cut_through])

    def test_a_q_limit_drops_the_bin_it_cuts(self):
        data = image(porod)
        full = data_ops.circular_average(data)
        edges = full.averaging['edges']
        qmin = (edges[10] + edges[11]) / 2
        qmax = (edges[20] + edges[21]) / 2
        cut = data_ops.circular_average(data, qmin=qmin, qmax=qmax)
        kept = np.flatnonzero(finite(cut))
        assert kept.min() == 11
        assert kept.max() == 19
        assert cut.averaging['qmin'] == pytest.approx(edges[11])

    def test_a_cut_as_bins_puts_another_image_on_its_grid(self):
        first = image(porod)
        shifted = image(porod, qmax=0.1 * (1 + 2e-3))
        a = data_ops.circular_average(first)
        assert data_ops.circular_average(shifted).x.size != a.x.size or not np.array_equal(
            data_ops.circular_average(shifted).x, a.x
        )
        b = data_ops.circular_average(shifted, bins=a)
        np.testing.assert_array_equal(b.x, a.x)
        difference = quietly(data_ops.subtract, b, a)
        assert difference.x.size == a.x.size

    def test_sanspol_channels_on_one_grid(self):
        plus = examples.load_fitter('sanspol_plus')
        minus = examples.load_fitter('sanspol_minus')
        across_plus = quietly(plus.sector_average, 'vertical')
        across_minus = quietly(minus.sector_average, 'vertical', bins=across_plus)
        np.testing.assert_array_equal(across_minus.x, across_plus.x)
        assert quietly(data_ops.subtract, across_minus, across_plus).x.size == across_plus.x.size

    def test_a_data1d_as_bins_gives_its_q_values(self):
        data = image(porod)
        q = np.geomspace(0.01, 0.09, 25)
        target = Data1D(x=q, y=np.ones_like(q))
        cut = data_ops.circular_average(data, bins=target)
        np.testing.assert_array_equal(cut.x, q)

    def test_a_coarse_curve_as_bins_keeps_its_fully_covered_bins(self):
        # Halfway edges around log-spaced Q values put the ring mean several
        # per cent from those values; that is not a bin the mask cut through.
        data = image(porod, n=128)
        q = 0.005 * 1.5 ** np.arange(7)
        cut = data_ops.circular_average(data, bins=Data1D(x=q, y=np.ones_like(q)))
        np.testing.assert_array_equal(cut.x, q)
        assert cut.averaging['edge_bins'].size == 0
        assert finite(cut).all()

    def test_default_grid_ignores_nearly_coincident_pixels(self):
        # Two overlapping panels: the neighbour distance is their offset.
        single = image(porod)
        qx = np.concatenate([single.qx_data, single.qx_data + 1e-6])
        qy = np.concatenate([single.qy_data, single.qy_data + 3e-7])
        double = Data2D(
            data=porod(qx, qy),
            err_data=np.full(qx.size, 0.01),
            qx_data=qx,
            qy_data=qy,
            q_data=np.hypot(qx, qy),
            mask=np.ones(qx.size, dtype=bool),
        )
        n_single = data_ops.circular_average(single).x.size
        n_double = data_ops.circular_average(double).x.size
        assert n_single <= n_double <= 2 * n_single


# =============================================================================
# Resolution
# =============================================================================


class TestResolution:
    def test_file_widths_give_dq_with_the_bin_spread(self):
        data = image(dq=0.05)
        cut = data_ops.circular_average(data)
        ok = finite(cut)
        sigma = 0.05 * np.hypot(data.qx_data, data.qy_data)
        index = (
            np.searchsorted(cut.averaging['edges'], np.hypot(data.qx_data, data.qy_data), 'right')
            - 1
        )
        for i in np.flatnonzero(ok)[::7]:
            pixels = index == i
            q = np.hypot(data.qx_data, data.qy_data)[pixels]
            expected = np.sqrt(np.mean(sigma[pixels] ** 2) + np.mean((q - cut.x[i]) ** 2))
            assert cut.dx[i] == pytest.approx(expected, rel=1e-10)

    def test_no_dq_in_the_image_gives_none(self):
        cut = data_ops.circular_average(image())
        assert cut.dx is None
        assert 'not in the file' in cut.averaging['resolution']

    def test_the_fitters_pinhole_setting_is_carried_over(self):
        fitter = fitter_for(image())
        fitter.set_resolution('pinhole', dq_over_q=0.08)
        cut = fitter.circular_average()
        ok = finite(cut)
        spread = cut.averaging['q_spread'][ok]
        # mean((0.08 q)²) over a bin is close to (0.08 Q_bin)²; the spread adds.
        np.testing.assert_allclose(
            cut.dx[ok] ** 2 - spread**2, (0.08 * cut.averaging['q_mean'][ok]) ** 2, rtol=0.05
        )
        fitter.set_resolution('none')
        assert fitter.circular_average().dx is None

    @pytest.mark.slow
    def test_a_1d_fit_with_the_cut_dq_recovers_the_radius(self):
        data = examples.simulate_2d(
            'sphere', qmax=0.15, npixels=128, noise=0.005, dq=0.1, radius=60, scale=0.01, seed=3
        )
        cut = quietly(data_ops.circular_average, data, bins=60)
        fitter = IQuana()
        fitter.set_data(cut)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=50, min=20, max=100, vary=True)
        fitter.set_param('scale', value=0.01, min=1e-4, max=1, vary=True)
        fitter.set_param('background', value=0.0, vary=False)
        result = quietly(fitter.fit, engine='lmfit')
        assert result['parameters']['radius']['value'] == pytest.approx(60, rel=0.02)


# =============================================================================
# Statistics
# =============================================================================


class TestStatistics:
    def test_pulls_have_unit_width(self):
        rng = np.random.default_rng(1)
        base = image(lambda qx, qy: 5.0 + 0 * qx, n=32, err=0.5)
        pulls = {'circular': [], 'sector': [], 'annular': []}
        for _ in range(200):
            noisy = image(lambda qx, qy: 5.0 + 0.5 * rng.standard_normal(qx.size), n=32, err=0.5)
            noisy.qx_data, noisy.qy_data = base.qx_data, base.qy_data
            circle = data_ops.circular_average(noisy, bins=8)
            sector = data_ops.sector_average(noisy, 0, half_width=30, bins=8)
            ring = data_ops.annular_average(noisy, 0.03, 0.09, bins=12)
            for name, values, sigma in (
                ('circular', circle.y, circle.dy),
                ('sector', sector.y, sector.dy),
                ('annular', ring.intensity, ring.uncertainty),
            ):
                ok = np.isfinite(values)
                pulls[name].extend((values[ok] - 5.0) / sigma[ok])
        for name, values in pulls.items():
            assert np.std(values) == pytest.approx(1.0, abs=0.2), name

    def test_scatter_uncertainty_matches_the_true_scatter(self):
        rng = np.random.default_rng(2)
        means = []
        errors = []
        for _ in range(200):
            noisy = image(lambda qx, qy: 5.0 + 0.5 * rng.standard_normal(qx.size), n=32, err=None)
            cut = quietly(data_ops.circular_average, noisy, bins=8)
            means.append(cut.y)
            errors.append(cut.dy)
        means, errors = np.array(means), np.array(errors)
        ok = np.all(np.isfinite(means), axis=0)
        np.testing.assert_allclose(
            np.nanmean(errors[:, ok], axis=0), np.std(means[:, ok], axis=0), rtol=0.2
        )

    def test_fit_sin2_pulls(self):
        rng = np.random.default_rng(3)
        truth = sin2(4.0, 1.0, 30.0)
        pulls = {'a': [], 'b': [], 'phi0': []}
        for _ in range(200):
            noisy = image(
                lambda qx, qy: truth(qx, qy) + 0.3 * rng.standard_normal(qx.size), n=48, err=0.3
            )
            fit = data_ops.annular_average(noisy, 0.03, 0.09).fit_sin2()
            assert fit.b / fit.b_err >= 10
            pulls['a'].append((fit.a - 4.0) / fit.a_err)
            pulls['b'].append((fit.b - 1.0) / fit.b_err)
            pulls['phi0'].append(((fit.phi0 - 30.0 + 90) % 180 - 90) / fit.phi0_err)
        for name, values in pulls.items():
            assert np.std(values) == pytest.approx(1.0, abs=0.2), name

    def test_isotropic_data_warn_weak_anisotropy(self):
        rng = np.random.default_rng(4)
        noisy = image(lambda qx, qy: 4.0 + 0.3 * rng.standard_normal(qx.size), n=48, err=0.3)
        ring = data_ops.annular_average(noisy, 0.03, 0.09)
        with pytest.warns(UserWarning, match='weak_anisotropy'):
            fit = ring.fit_sin2()
        assert np.isnan(fit.phi0_err)
        assert 'weak_anisotropy' in fit.warning_codes


# =============================================================================
# Behaviour
# =============================================================================


class TestAngles:
    @pytest.mark.parametrize(('phi', 'same'), [(0, 360), (370, 10), (-30, 330), (355, -5)])
    def test_equivalent_angles_select_the_same_pixels(self, phi, same):
        data = image(porod)
        a = data_ops.sector_average(data, phi, mirror=False)
        b = data_ops.sector_average(data, same, mirror=False)
        np.testing.assert_array_equal(a.averaging['pixels'], b.averaging['pixels'])

    def test_named_directions(self):
        data = image(porod)
        np.testing.assert_array_equal(
            data_ops.sector_average(data, 'vertical').averaging['pixels'],
            data_ops.sector_average(data, 90).averaging['pixels'],
        )
        assert data_ops.sector_average(data, 'Horizontal').averaging['phi'] == 0.0

    def test_mirror_pools_both_sides(self):
        data = image(porod)
        one = data_ops.sector_average(data, 0, mirror=False).averaging['n_used']
        other = data_ops.sector_average(data, 180, mirror=False).averaging['n_used']
        assert data_ops.sector_average(data, 0).averaging['n_used'] == one + other

    def test_a_detector_covering_one_side_warns_one_sided(self):
        data = image(porod, keep=lambda qx, qy: qx > 0.005)
        with pytest.warns(UserWarning, match='one_sided'):
            cut = data_ops.sector_average(data, 0)
        assert finite(cut).any()

    def test_a_shifted_beam_centre_warns_sides_differ(self):
        def shifted(qx, qy):
            return porod(qx - 0.003, qy)

        data = image(shifted, err=0.01, relative=True, keep=lambda qx, qy: np.hypot(qx, qy) > 0.01)
        with pytest.warns(UserWarning, match='sides_differ'):
            data_ops.sector_average(data, 0)

    def test_a_centred_image_does_not_warn_even_with_tiny_errors_and_a_gradient(self):
        def gradient(qx, qy):
            return porod(qx, qy) * (1 + 0.005 * qx / 0.1)

        data = image(gradient, err=1e-4, relative=True, keep=lambda qx, qy: np.hypot(qx, qy) > 0.01)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            data_ops.sector_average(data, 0)
            data_ops.sector_average(image(porod, err=0.01, relative=True), 0)


class TestPixels:
    def test_both_mask_conventions_select_what_set_data_selects(self):
        data = image(porod)
        data.mask = np.hypot(data.qx_data, data.qy_data) > 0.02  # sasdata: True = included
        included = data_ops.circular_average(data)
        fitter = fitter_for(data)
        assert included.averaging['n_used'] == fitter.get_mask()['fitted']
        flipped = image(porod)
        flipped.mask = ~data.mask
        excluded = data_ops.circular_average(flipped, mask_convention='exclude')
        np.testing.assert_array_equal(excluded.averaging['pixels'], included.averaging['pixels'])

    def test_fitter_methods_honour_every_mask_and_the_q_range(self):
        data = image(porod)
        data.err_data[::97] = 0.0
        fitter = fitter_for(data)
        fitter.mask_ring(qmax=0.015)
        fitter.mask_box(qx=(-0.005, 0.005))
        fitter.mask_invalid()
        fitter.set_q_range(0.02, 0.08)
        cut = fitter.circular_average()
        fitted = fitter._fit_index()
        assert cut.averaging['n_used'] == int(fitted.sum())
        np.testing.assert_array_equal(
            np.unpackbits(cut.averaging['pixels'], count=fitted.size).astype(bool), fitted
        )
        fitter.set_mask(lambda qx, qy: qy > 0)
        assert fitter.circular_average().averaging['n_used'] == int(fitter._fit_index().sum())

    def test_mask_array_and_callable(self):
        data = image(porod)
        array = data.qx_data > 0.05
        by_array = data_ops.circular_average(data, mask=array)
        by_callable = data_ops.circular_average(data, mask=lambda qx, qy: qx > 0.05)
        np.testing.assert_array_equal(by_array.averaging['pixels'], by_callable.averaging['pixels'])
        assert by_array.averaging['n_used'] == int((~array).sum())

    def test_zero_uncertainty_pixels_are_kept_and_reported(self):
        data = image(err=0.1)
        data.err_data[::50] = 0.0
        with pytest.warns(UserWarning, match='zero_uncertainty_pixels'):
            cut = data_ops.circular_average(data)
        assert cut.averaging['n_used'] == data.data.size
        fitter = fitter_for(data)
        fitter.mask_invalid()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            masked = fitter.circular_average()
        assert masked.averaging['n_used'] == data.data.size - len(data.err_data[::50])

    def test_an_all_zero_di_column_counts_as_no_di(self):
        data = image(err=None)  # zero-filled, as sasdata fills an absent column
        with pytest.warns(UserWarning, match='uncertainties_from_scatter'):
            cut = data_ops.circular_average(data, bins=10)
        assert np.all(np.isfinite(cut.dy[finite(cut)]))

    def test_without_di_small_bins_are_nan_and_masked(self):
        data = image(lambda qx, qy: 1 + 0.1 * np.sin(97 * qx), err=None)
        cut = quietly(data_ops.circular_average, data)
        small = cut.averaging['n_pixels'] < MIN_SCATTER_PIXELS
        assert small.any()
        assert np.all(np.isnan(cut.y[small]))
        fitter = fitter_for(cut)
        assert np.all(fitter.data.mask[small])

    def test_without_di_and_no_bin_of_five_pixels_raises(self):
        data = image(err=None, n=8)
        with pytest.raises(ValueError, match='coarser bins'):
            quietly(data_ops.sector_average, data, 45, half_width=1, bins=np.linspace(0, 0.2, 200))

    def test_a_ring_without_di_and_no_bin_of_five_pixels_raises(self):
        data = image(porod, err=None, n=16)
        with pytest.raises(ValueError, match='fewer angular bins'):
            quietly(data_ops.annular_average, data, 0.03, 0.05, bins=72)

    def test_the_input_is_never_modified(self):
        data = image(porod, dq=0.05)
        before = {
            name: np.array(getattr(data, name), copy=True)
            for name in ('data', 'err_data', 'mask', 'qx_data')
        }
        data_ops.circular_average(data, mask=lambda qx, qy: qx > 0)
        data_ops.sector_average(data, 0)
        data_ops.annular_average(data, 0.02, 0.06)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(data, name), values)


class TestUsingTheCuts:
    def test_two_sectors_of_one_image_subtract(self):
        data = image(sin2(2, 1, 0))
        across = data_ops.sector_average(data, 90)
        along = data_ops.sector_average(data, 0)
        np.testing.assert_array_equal(across.x, along.x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            difference = data_ops.subtract(across, along)
        ok = np.isfinite(difference.y) & (difference.x < 0.1)  # inside the inscribed circle
        assert ok.any()
        np.testing.assert_allclose(difference.y[ok], 0.98, atol=0.02)

    def test_cuts_sharing_pixels_warn(self):
        data = image(porod)
        circle = data_ops.circular_average(data)
        along = data_ops.sector_average(data, 0)
        across = data_ops.sector_average(data, 90)
        with pytest.warns(UserWarning, match='shared_pixels'):
            data_ops.subtract(circle, along)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data_ops.subtract(across, along)
        assert not any('shared_pixels' in str(w.message) for w in caught)

    def test_dq_on_the_second_operand_only_is_a_clear_error(self):
        without = data_ops.circular_average(image(porod))
        with_dq = data_ops.circular_average(image(porod, dq=0.05))
        with pytest.raises(ValueError, match='second operand has dQ'):
            data_ops.subtract(without, with_dq)

    def test_empty_bins_are_masked_and_the_q_range_is_that_of_the_data(self):
        cut = data_ops.sector_average(image(porod), 0, half_width=5)
        assert np.isnan(cut.y).any()
        fitter = fitter_for(cut)
        assert np.all(fitter.data.mask[np.isnan(cut.y)])
        assert fitter.get_q_range() == (cut.x[finite(cut)].min(), cut.x[finite(cut)].max())

    def test_a_fit_to_a_cut_saves_and_loads_without_data(self, tmp_path):
        data = examples.simulate_2d('sphere', qmax=0.15, npixels=64, radius=40, scale=0.05, seed=1)
        cut = quietly(data_ops.circular_average, data)
        fitter = fitter_for(cut)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=35, min=10, max=100, vary=True)
        quietly(fitter.fit)
        path = tmp_path / 'cut.json'
        fitter.save_analysis(str(path))
        loaded = quietly(IQuana.load_analysis, str(path))
        np.testing.assert_array_equal(loaded.data.x, cut.x)
        assert any('Circular average' in line for line in loaded._data_source.processes)
        assert 'Circular average' in quietly(fitter.report, fmt='markdown').to_html()

    def test_a_cut_is_accepted_by_the_1d_analyses(self):
        data = examples.simulate_2d('sphere', qmax=0.3, npixels=96, radius=30, scale=0.05, seed=2)
        cut = quietly(data_ops.circular_average, data)
        multi = MultiFitter()
        multi.add('cut', cut, model='sphere')
        assert quietly(modelfree.guinier, cut).rg > 0
        assert quietly(modelfree.invariant, cut).qstar > 0
        assert quietly(pr_inversion.invert, cut, d_max=80, n_terms=10) is not None

    def test_an_annular_average_is_refused_by_the_fitters_and_analyses(self):
        ring = data_ops.annular_average(image(), 0.02, 0.06)
        assert isinstance(ring, AnnularAverage)
        for call in (
            lambda: IQuana().set_data(ring),
            lambda: modelfree.guinier(ring),
            lambda: pr_inversion.invert(ring, d_max=80),
        ):
            with pytest.raises(TypeError, match='intensity against angle'):
                call()
        with pytest.raises(TypeError, match='annular average'):
            data_ops.subtract(ring, ring)

    def test_the_fitter_ring_ignores_the_q_range_and_keeps_the_masks(self):
        fitter = fitter_for(image(porod))
        fitter.set_q_range(0.05, 0.08)
        fitter.mask_box(qx=(0.0, 1.0))
        ring = fitter.annular_average(0.02, 0.04)
        q = np.hypot(fitter.data.qx_data, fitter.data.qy_data)
        expected = (q >= 0.02) & (q <= 0.04) & (fitter.data.qx_data < 0.0)
        assert ring.n_used == int(expected.sum())

    def test_uneven_ring_warns_past_the_detector_edge(self):
        data = image(porod)
        with pytest.warns(UserWarning, match='uneven_ring'):
            data_ops.annular_average(data, 0.07, 0.13)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            data_ops.annular_average(data, 0.04, 0.06)

    def test_partial_ring(self):
        data = image(keep=lambda qx, qy: (qx > 0) & (qy > 0))
        with pytest.warns(UserWarning, match='partial_ring'):
            ring = data_ops.annular_average(data, 0.02, 0.06)
        assert np.isnan(ring.intensity).sum() > ring.phi.size / 2

    def test_ring_summary_csv_and_report(self, tmp_path):
        ring = data_ops.annular_average(image(sin2(2, 1, 0), n=128), 0.03, 0.07)
        assert 'Annular average' in ring.format_summary()
        path = tmp_path / 'ring.csv'
        ring.save_csv(str(path))
        rows = [line for line in path.read_text().splitlines() if not line.startswith('#')]
        assert rows[0] == 'phi_deg,I,dI,n_pixels,q_mean'
        assert len(rows) == 37
        fit = ring.fit_sin2()
        report = fit.report()
        assert report.model == 'sin2'
        assert set(report.parameters) == {'a', 'b', 'phi0'}
        assert 'least intensity' in fit.format_summary()


class TestErrors:
    def test_1d_data_is_refused(self):
        curve = examples.simulate('sphere', noise=0)
        for call in (data_ops.circular_average, lambda d: data_ops.sector_average(d, 0)):
            with pytest.raises(TypeError, match='2D detector image'):
                call(curve)
        with pytest.raises(ValueError, match='needs 2D data'):
            fitter_for(curve).circular_average()

    def test_an_empty_region_lists_the_stages(self):
        data = image(keep=lambda qx, qy: qx > 0.01)
        with pytest.raises(ValueError, match='degrees, anticlockwise from \\+Qx'):
            data_ops.sector_average(data, 180, mirror=False)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'half_width': 0},
            {'half_width': 91},
            {'half_width': 181, 'mirror': False},
            {'phi': float('nan')},
            {'phi': 'diagonal'},
            {'qmin': 0.05, 'qmax': 0.02},
            {'bins': 0},
            {'bins': [0.01, 0.005]},
            {'bins': [0.0, np.inf]},
            {'bins': [[0.0, 0.1]]},
            {'bins': [0.0, 0.05, 0.1], 'log': True},
            {'log': True},
        ],
    )
    def test_argument_rules(self, kwargs):
        arguments = {'phi': 0, **kwargs}
        with pytest.raises(ValueError):
            data_ops.sector_average(image(), **arguments)

    def test_a_cut_as_bins_with_log_is_refused(self):
        data = image()
        with pytest.raises(ValueError, match='log=True'):
            data_ops.circular_average(data, bins=data_ops.circular_average(data), log=True)

    def test_half_width_90_mirrored_is_the_full_circle(self):
        data = image(porod)
        sector = data_ops.sector_average(data, 0, half_width=90)
        np.testing.assert_array_equal(sector.y, data_ops.circular_average(data).y)


class TestOutput:
    def test_log_line_and_quiet(self):
        data = image(porod)
        set_verbosity('info')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            data_ops.sector_average(data, 90)
            fitter_for(data).sector_average(90)
        text = buffer.getvalue()
        assert 'Sector average' in text and '(mirrored)' in text
        set_verbosity('quiet')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            data_ops.circular_average(data)
            fitter = fitter_for(data)
            fitter.circular_average()
            fitter.annular_average(0.02, 0.05)
        assert buffer.getvalue() == ''

    def test_the_cut_carries_metadata_and_provenance(self):
        data = examples.load('sanspol_plus')
        cut = quietly(data_ops.sector_average, data, 90)
        assert cut.title == 'S50_I+_2D.dat sector 90°±10°'
        assert cut.process[-1].name == 'iquana.data_ops'
        assert 'Sector average' in cut.process[-1].description
        assert cut.process[-1].description.isascii()
        assert cut._yunit == data._zunit

    def test_one_megapixel_image_is_fast(self):
        data = image(porod, n=1024)
        fitter = fitter_for(data)
        start = time.perf_counter()
        fitter.circular_average()
        first = time.perf_counter() - start
        start = time.perf_counter()
        fitter.sector_average(90)
        second = time.perf_counter() - start
        assert first < 5.0
        assert second < 0.5


class TestPlotCuts:
    def test_one_outline_per_region_two_for_a_mirrored_sector(self):
        data = image(porod)
        fitter = fitter_for(data)
        circle = fitter.circular_average()
        mirrored = fitter.sector_average(0)
        single = fitter.sector_average(90, mirror=False)
        ring = fitter.annular_average(0.02, 0.05)
        fig = fitter.plot_cuts(circle, mirrored, single, ring, show=False)
        outlines = [t for t in fig.data if t.type == 'scatter' and t.showlegend is False]
        assert len(outlines) == 2 + 2 + 1 + 2
        assert fig.layout.yaxis.autorange != 'reversed'
        assert [t.name for t in fig.data if t.type == 'scatter' and t.showlegend is not False] == [
            circle.title,
            mirrored.title,
            single.title,
            ring.title,
        ]

    def test_a_cut_without_its_record_or_from_another_image_warns(self):
        data = image(porod)
        cut = data_ops.circular_average(data)
        plain = Data1D(x=cut.x, y=cut.y)
        with pytest.warns(UserWarning, match='no averaging record'):
            data_ops.plot_cuts(data, plain, show=False)
        other = data_ops.circular_average(image(porod, qmax=0.12))
        with pytest.warns(UserWarning, match='not made from this image'):
            data_ops.plot_cuts(data, other, show=False)

    def test_ring_and_fit_plots(self):
        ring = data_ops.annular_average(image(sin2(), n=64), 0.03, 0.07)
        assert len(ring.plot(show=False).data) == 1
        assert len(ring.fit_sin2().plot(show=False).data) == 2
