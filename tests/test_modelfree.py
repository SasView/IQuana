"""Tests for iquana.modelfree: Guinier, power law, Porod, Kratky and the invariant.

Oracles come from analytic results, not from the implementation's formulas.
Every simulation passes ``background=0`` unless it tests the background, and
noise-free data gets small explicit uncertainties, since ``simulate(noise=0)``
returns none.
"""

import json
import math
import warnings

import numpy as np
import pytest
from sasdata.dataloader.data_info import Data1D
from scipy.integrate import quad

from iquana import examples, modelfree
from iquana.data.loader import normalize_sans_data
from iquana.modelfree import InsufficientDataError, ModelFreeError

R = 50.0
RG = math.sqrt(0.6) * R


@pytest.fixture(autouse=True)
def quiet():
    """The analyses warn by design; tests that check a warning read result.warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def sphere(noise=0.01, seed=0, npoints=200, qmin=0.003, qmax=0.5, **params):
    params = {'radius': R, 'background': 0, **params}
    return examples.simulate(
        'sphere', qmin=qmin, qmax=qmax, npoints=npoints, noise=noise, seed=seed, **params
    )


def noise_free(model, **kwargs):
    data = examples.simulate(model, noise=0, background=0, **kwargs)
    data.dy = 1e-4 * np.abs(data.y)
    return data


def make_data(q, intensity, sigma=None):
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    dy = np.zeros_like(q) if sigma is None else np.asarray(sigma, dtype=float)
    return normalize_sans_data(Data1D(x=q, y=intensity, dy=dy))


def polydisperse_spheres(seed=0, noise=0.01):
    return examples.simulate(
        'sphere',
        radius=R,
        radius_pd=0.2,
        qmin=0.002,
        qmax=0.6,
        npoints=300,
        background=0.001,
        noise=noise,
        sld=4,
        sld_solvent=1,
        scale=0.01,
        seed=seed,
    )


def sv_oracle(phi=0.01, width=0.2):
    """3φ⟨R²⟩/⟨R³⟩ over the discretised Gaussian simulate() uses (35 points, ±3σ)."""
    from sasmodels.weights import get_weights

    radii, weights = get_weights('gaussian', 35, width, 3, R, (0.0, math.inf), True)
    return 3 * phi * np.sum(weights * radii**2) / np.sum(weights * radii**3)


# =============================================================================
# Recovery
# =============================================================================


def test_guinier_sphere_auto_window():
    g = modelfree.guinier(sphere())
    assert g.window_selected == 'auto'
    assert g.qr_max <= 1.3
    assert g.rg == pytest.approx(RG, rel=0.03)
    assert g.sphere_radius == pytest.approx(R, rel=0.03)
    assert g.i0 == pytest.approx(sphere(noise=0).y[0], rel=0.02)


def test_guinier_matches_noise_free_fit_over_same_window():
    clean = modelfree.guinier(noise_free('sphere', radius=R, qmin=0.003, qmax=0.5), qmax=0.03)
    noisy = modelfree.guinier(sphere(), qmax=0.03)
    assert abs(noisy.rg - clean.rg) < 3 * noisy.rg_err


def test_guinier_rod_cross_section():
    data = noise_free('cylinder', radius=20, length=400, qmin=0.001, qmax=0.3, npoints=400)
    g = modelfree.guinier(data, kind='rod', qmin=1.5 * 2 * math.pi / 400)
    assert g.radius == pytest.approx(20 / math.sqrt(2), rel=0.05)
    assert g.cross_section_radius == pytest.approx(20, rel=0.05)


def test_guinier_sheet_thickness():
    data = noise_free('cylinder', radius=400, length=20, qmin=0.001, qmax=0.5, npoints=400)
    g = modelfree.guinier(data, kind='sheet', qmin=1.5 * 2 * math.pi / 800)
    assert g.thickness == pytest.approx(20, rel=0.05)


@pytest.mark.parametrize('fit_background', [False, True])
def test_power_law_exact(fit_background):
    q = np.geomspace(0.01, 0.3, 60)
    intensity = 2e-3 * q**-2.5 + 0.05
    data = make_data(q, intensity, 1e-4 * intensity)
    kwargs = {'fit_background': True} if fit_background else {'background': 0.05}
    pl = modelfree.power_law(data, qmin=0.01, qmax=0.3, **kwargs)
    assert pl.exponent == pytest.approx(2.5, rel=1e-4)
    assert pl.prefactor == pytest.approx(2e-3, rel=1e-3)
    assert pl.background == pytest.approx(0.05, rel=1e-3)
    assert pl.background_fitted is fit_background


def test_porod_specific_surface_absolute_scale():
    data = noise_free(
        'sphere',
        radius=R,
        radius_pd=0.2,
        qmin=0.002,
        qmax=0.6,
        npoints=300,
        sld=4,
        sld_solvent=1,
        scale=0.01,
    )
    p = modelfree.porod(data, qmin=0.15)
    assert p.free_exponent == pytest.approx(4.0, abs=0.05)
    assert 'porod_exponent' not in p.warning_codes
    assert p.specific_surface(3.0) == pytest.approx(sv_oracle(), rel=0.02)


def test_invariant_sphere_oracle():
    # Polydisperse, so the Porod tail is smooth; Q* does not depend on the size distribution.
    data = noise_free(
        'sphere',
        radius=R,
        radius_pd=0.2,
        qmin=0.001,
        qmax=1.0,
        npoints=600,
        sld=4,
        sld_solvent=1,
        scale=0.01,
    )
    p = modelfree.porod(data, qmin=0.3)
    g = modelfree.guinier(data, background=p.background)
    inv = modelfree.invariant(data, low_q=g, high_q=p, delta_sld=3)
    assert inv.qstar == pytest.approx(2 * math.pi**2 * 3**2 * 0.01 * 1e-4, rel=0.01)
    assert inv.phi_product == pytest.approx(0.01, rel=0.01)
    assert inv.qstar == pytest.approx(inv.low_q_part + inv.measured_part + inv.high_q_part)


def test_extrapolations_match_quadrature():
    data = sphere(noise=0.01)
    g = modelfree.guinier(data)
    inv = modelfree.invariant(data, low_q=g)
    expected, _ = quad(lambda q: q**2 * g.i0 * math.exp(-((q * g.rg) ** 2) / 3), 0, inv.qmin)
    assert inv.low_q_part == pytest.approx(expected, rel=1e-8)

    # Flat at low Q (n near 0 < 3), q^-4 at high Q (n near 4 > 3).
    q = np.geomspace(0.002, 0.5, 120)
    intensity = 1.0 / (1.0 + (q / 0.05) ** 4)
    curve = make_data(q, intensity, 1e-3 * intensity)
    low_fit = modelfree.power_law(curve, qmin=0.002, qmax=0.01)
    high_fit = modelfree.power_law(curve, qmin=0.3, qmax=0.5)
    inv = modelfree.invariant(curve, low_q=low_fit, high_q=high_fit)
    a, n = low_fit.prefactor, low_fit.exponent
    assert inv.low_q_part == pytest.approx(
        quad(lambda x: a * x ** (2 - n), 0, inv.qmin)[0], rel=1e-6
    )
    a, n = high_fit.prefactor, high_fit.exponent
    assert inv.high_q_part == pytest.approx(
        quad(lambda x: a * x ** (2 - n), inv.qmax, np.inf, epsabs=0, epsrel=1e-11)[0], rel=1e-6
    )


def test_kratky_gaussian_peak_is_exact():
    q = np.linspace(0.001, 0.3, 3000)
    data = make_data(q, 100 * np.exp(-((q * 30) ** 2) / 3), np.full(q.size, 1e-6))
    g = modelfree.guinier(data, qr_limit=1.0)
    k = modelfree.kratky(data, guinier=g)
    assert k.peak_position == pytest.approx(math.sqrt(3), rel=1e-4)
    assert k.peak_height == pytest.approx(3 / math.e, rel=1e-4)


def test_kratky_sphere_peak():
    data = noise_free('sphere', radius=R, qmin=0.001, qmax=0.2, npoints=2000)
    g = modelfree.guinier(data, qr_limit=0.6)
    k = modelfree.kratky(data, guinier=g)
    assert k.dimensionless
    assert k.peak_position == pytest.approx(1.612, rel=0.01)
    assert k.peak_height == pytest.approx(1.027, rel=0.01)


def test_kratky_rising_curve_has_no_peak():
    # A Gaussian coil's dimensionless Kratky curve rises to a plateau of 2: no peak.
    data = noise_free('mono_gauss_coil', i_zero=1, rg=RG, qmin=0.001, qmax=0.5, npoints=400)
    k = modelfree.kratky(data, guinier=modelfree.guinier(data, qr_limit=1.0))
    assert k.peak_position is None and k.peak_height is None


def test_kratky_plain_form():
    data = sphere()
    k = modelfree.kratky(data, background=0.0)
    assert not k.dimensionless
    np.testing.assert_allclose(k.y, k.q**2 * k.intensity)


# =============================================================================
# Statistics
# =============================================================================


def test_guinier_pull_width():
    clean = modelfree.guinier(noise_free('sphere', radius=R, qmin=0.003, qmax=0.5), qmax=0.03)
    pulls = [
        (g.rg - clean.rg) / g.rg_err
        for g in (modelfree.guinier(sphere(seed=seed), qmax=0.03) for seed in range(100))
    ]
    assert np.std(pulls) == pytest.approx(1.0, abs=0.2)


def test_invariant_error_matches_scatter():
    values, errors = [], []
    for seed in range(100):
        data = polydisperse_spheres(seed=seed)
        p = modelfree.porod(data, qmin=0.15)
        g = modelfree.guinier(data, background=p.background)
        inv = modelfree.invariant(data, low_q=g, high_q=p, delta_sld=3, volume_fraction=0.01)
        values.append([inv.qstar, inv.specific_surface_porod, inv.specific_surface_invariant])
        errors.append(
            [inv.qstar_err, inv.specific_surface_porod_err, inv.specific_surface_invariant_err]
        )
    ratio = np.std(values, axis=0) / np.mean(errors, axis=0)
    np.testing.assert_allclose(ratio, 1.0, atol=0.2)


@pytest.mark.parametrize(
    ('model', 'params', 'expected'),
    [
        ('sphere', {'radius': R}, None),
        ('mono_gauss_coil', {'i_zero': 1, 'rg': RG}, 'curvature_positive'),
        ('cylinder', {'radius': 3, 'length': 134}, 'curvature_positive'),
        (
            'sphere@hardsphere',
            {'radius': R, 'radius_effective': R, 'volfraction': 0.05},
            'curvature_negative',
        ),
    ],
)
def test_curvature_check(model, params, expected):
    for seed in range(10):
        data = examples.simulate(
            model, qmin=0.003, qmax=0.5, npoints=300, noise=0.002, background=0, seed=seed, **params
        )
        codes = modelfree.guinier(data).warning_codes
        curvature = {c for c in codes if c.startswith('curvature')}
        assert curvature == ({expected} if expected else set())


# =============================================================================
# Checks and refusals
# =============================================================================


def test_guinier_region_not_measured():
    with pytest.raises(ModelFreeError, match='smallest value reached'):
        modelfree.guinier(sphere(qmin=0.05))


@pytest.mark.parametrize('kind', ['rod', 'sheet'])
def test_rod_and_sheet_need_qmin(kind):
    with pytest.raises(ModelFreeError, match='qmin'):
        modelfree.guinier(sphere(), kind=kind)


def test_guinier_explicit_window_warns_past_limit():
    g = modelfree.guinier(sphere(), qmax=0.05)
    assert g.window_selected == 'user'
    assert 'qr_limit' in g.warning_codes


def test_guinier_low_q_coverage_and_few_points():
    g = modelfree.guinier(sphere(qmin=0.028, npoints=30), qr_limit=1.6, min_points=3)
    assert 'low_q_coverage' in g.warning_codes
    assert 'few_points' in g.warning_codes
    assert g.curvature is None


def test_kind_specific_properties():
    g = modelfree.guinier(sphere())
    with pytest.raises(ModelFreeError):
        _ = g.thickness
    data = noise_free('cylinder', radius=20, length=400, qmin=0.001, qmax=0.3, npoints=400)
    rod = modelfree.guinier(data, kind='rod', qmin=0.025)
    with pytest.raises(ModelFreeError):
        _ = rod.rg
    with pytest.raises(ModelFreeError):
        modelfree.kratky(data, guinier=rod)
    with pytest.raises(ModelFreeError):
        modelfree.invariant(data, low_q=rod)


def test_power_law_low_signal_and_short_range():
    q = np.geomspace(0.1, 0.15, 30)
    rng = np.random.default_rng(0)
    intensity = 0.01 + 1e-6 * q**-4 + rng.normal(0, 0.005, q.size)
    data = make_data(q, intensity, np.full(q.size, 0.005))
    pl = modelfree.power_law(data, qmin=0.1, qmax=0.15, background=0.01)
    assert 'low_signal' in pl.warning_codes
    assert 'short_q_range' in pl.warning_codes


def test_power_law_free_fit_reports_optimizer_and_fails_on_noise():
    q = np.geomspace(0.01, 0.3, 60)
    intensity = 2e-3 * q**-2.5 + 0.05
    pl = modelfree.power_law(
        make_data(q, intensity, 1e-3 * intensity), 0.01, 0.3, fit_background=True
    )
    report = pl.report()
    assert report.method == 'least_squares (trf)'
    assert report.cov_source == 'jacobian'
    assert report.message == pl.message and report.message != 'closed-form solution'

    noise = make_data(
        q, 1.0 + np.random.default_rng(1).normal(0, 0.01, q.size), np.full(q.size, 0.01)
    )
    with pytest.raises(ModelFreeError):
        modelfree.power_law(noise, 0.01, 0.3, fit_background=True)


def test_porod_negative_constant_and_exponent_warning():
    q = np.geomspace(0.1, 0.5, 40)
    rising = make_data(q, 1 + q**2, np.full(q.size, 0.01))
    with pytest.raises(ModelFreeError, match='negative'):
        modelfree.porod(rising, qmin=0.1)
    fractal = make_data(q, 1e-3 * q**-3 + 0.01, 1e-4 * (1e-3 * q**-3 + 0.01))
    assert 'porod_exponent' in modelfree.porod(fractal, qmin=0.1).warning_codes


def test_delta_sld_in_wrong_units():
    data = polydisperse_spheres()
    p = modelfree.porod(data, qmin=0.15)
    with pytest.raises(ValueError, match='1e-6'):
        modelfree.invariant(data, high_q=p, delta_sld=3e-6)
    with pytest.raises(ValueError, match='1e-6'):
        p.specific_surface(3e-6)


def test_invariant_refusals():
    data = polydisperse_spheres()
    p = modelfree.porod(data, qmin=0.15)
    with pytest.raises(ModelFreeError, match='<= 0'):
        modelfree.invariant(data, background=10.0)
    q = np.geomspace(0.01, 0.5, 60)
    steep = make_data(q, 1e-6 * q**-4, 1e-3 * 1e-6 * q**-4)
    low = modelfree.power_law(steep, qmin=0.01, qmax=0.05)
    with pytest.raises(ModelFreeError, match='diverge'):
        modelfree.invariant(steep, low_q=low)
    shallow = make_data(q, 1e-3 * q**-2, 1e-3 * 1e-3 * q**-2)
    high = modelfree.power_law(shallow, qmin=0.1, qmax=0.5)
    with pytest.raises(ModelFreeError, match='diverge'):
        modelfree.invariant(shallow, high_q=high)
    with pytest.raises(ModelFreeError, match='exceeds 1/4'):
        modelfree.invariant(data, high_q=p, delta_sld=0.01)


def test_background_mismatch_raises():
    data = polydisperse_spheres()
    p = modelfree.porod(data, qmin=0.15)
    g = modelfree.guinier(data)  # background 0, not Porod's
    with pytest.raises(ModelFreeError, match='Background mismatch'):
        modelfree.invariant(data, low_q=g, high_q=p)
    with pytest.raises(ModelFreeError, match='Background mismatch'):
        modelfree.kratky(data, guinier=g, background=p.background)


def test_fixed_background_inherited():
    q = np.geomspace(0.01, 0.5, 80)
    data = make_data(q, 1e-6 * q**-4 + 0.02, 1e-3 * (1e-6 * q**-4 + 0.02))
    high = modelfree.power_law(data, qmin=0.2, qmax=0.5, background=0.02)
    inv = modelfree.invariant(data, high_q=high)
    assert inv.background == 0.02
    assert 'background_uncertainty_excluded' in inv.warning_codes


def test_results_from_another_dataset_refused():
    p = modelfree.porod(polydisperse_spheres(), qmin=0.15)
    other = sphere(qmax=0.6)
    with pytest.raises(ModelFreeError, match='different dataset'):
        modelfree.invariant(other, high_q=p)


def test_window_not_at_edge():
    data = polydisperse_spheres()
    p = modelfree.porod(data, qmin=0.15, qmax=0.4)
    g = modelfree.guinier(data, qmin=0.004, background=p.background)
    inv = modelfree.invariant(data, low_q=g, high_q=p)
    assert inv.warning_codes.count('window_not_at_edge') == 2
    assert 'qmin=0.004' in ' '.join(message for _, message in inv.warnings)


def test_invariant_joins_at_measured_points():
    data = polydisperse_spheres()
    p = modelfree.porod(data, qmin=0.15)
    g = modelfree.guinier(data, background=p.background)
    inv = modelfree.invariant(data, low_q=g, high_q=p, qmin=0.0001, qmax=0.55)
    assert 'outside_dataset_range' in inv.warning_codes
    inside = data.x[(data.x <= 0.55)]
    assert inv.qmin == data.x[0] and inv.qmax == inside[-1]
    assert inv.qstar == pytest.approx(inv.low_q_part + inv.measured_part + inv.high_q_part)
    assert inv.measured_part == pytest.approx(
        np.trapezoid(inv.q**2 * (inv.intensity - inv.background), inv.q)
    )


def test_unsorted_and_repeated_q():
    data = sphere()
    order = np.random.default_rng(3).permutation(data.x.size)
    shuffled = make_data(data.x[order], data.y[order], data.dy[order])
    a, b = modelfree.guinier(data), modelfree.guinier(shuffled)
    assert b.rg == pytest.approx(a.rg, rel=1e-12)
    assert modelfree.invariant(shuffled).qstar == pytest.approx(modelfree.invariant(data).qstar)
    repeated = make_data(np.repeat(data.x, 2), np.repeat(data.y, 2), np.repeat(data.dy, 2))
    assert modelfree.invariant(repeated).measured_part == pytest.approx(
        modelfree.invariant(data).measured_part
    )


def test_too_few_points():
    q = np.geomspace(0.01, 0.02, 4)
    data = make_data(q, 100 * np.exp(-((q * 10) ** 2) / 3), np.full(4, 0.1))
    with pytest.raises(InsufficientDataError):
        modelfree.guinier(data)
    with pytest.raises(InsufficientDataError):
        modelfree.power_law(make_data(q[:2], [1.0, 0.5], [0.1, 0.1]), qmin=0.01, qmax=0.02)
    with pytest.raises(InsufficientDataError):
        modelfree.power_law(data, qmin=0.01, qmax=0.012, fit_background=True)
    negative = make_data(q, [1.0, -1.0, -1.0, 0.5], np.full(4, 0.1))
    with pytest.raises(InsufficientDataError):
        modelfree.power_law(negative, qmin=0.01, qmax=0.02)


@pytest.mark.parametrize(
    'call',
    [
        lambda d: modelfree.guinier(d),
        lambda d: modelfree.power_law(d, 0.01, 0.2),
        lambda d: modelfree.porod(d, qmin=0.1),
        lambda d: modelfree.kratky(d),
        lambda d: modelfree.invariant(d),
    ],
)
def test_slit_smeared_data_refused(call):
    data = sphere()
    data.dxl = np.full(data.x.size, 0.05)
    with pytest.raises(ValueError, match='slit-smeared'):
        call(data)


def test_2d_and_sesans_refused():
    data = sphere()
    data.isSesans = True
    with pytest.raises(TypeError, match='SESANS'):
        modelfree.guinier(data)
    with pytest.raises(TypeError, match='1D data only'):
        modelfree.kratky(examples.simulate_2d('sphere', noise=0))


def test_mask_q_range_and_missing_di():
    data = sphere()
    data.qmin = 0.006
    g = modelfree.guinier(data)
    assert g.qmin >= 0.006
    data.mask = np.zeros(data.x.size, dtype=bool)
    data.mask[g.index[0]] = True
    assert modelfree.guinier(data).qmin > g.qmin

    no_di = sphere()
    no_di.dy = np.zeros_like(no_di.dy)
    g = modelfree.guinier(no_di)
    assert g.uncertainties_fabricated
    assert 'uncertainties_fabricated' in g.warning_codes
    assert 'fabricated' in g.report().weighting_note


def test_dq_ignored():
    data = examples.simulate('sphere', radius=R, background=0, dq=0.05, qmin=0.003)
    assert 'dq_ignored' in modelfree.guinier(data).warning_codes


def test_nonpositive_points_counted_within_window():
    data = sphere()
    data.y[3] = -1.0
    g = modelfree.guinier(data)
    assert g.n_nonpositive_points == 1
    assert g.n_points + 1 == np.count_nonzero(data.x <= g.qmax)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'qmin': 0.02, 'qmax': 0.01}, 'smaller'),
        ({'qmin': -1}, 'positive'),
        ({'min_points': 2}, 'min_points'),
        ({'qr_limit': 0}, 'qr_limit'),
        ({'kind': 'globular'}, 'kind'),
        ({'background': math.nan}, 'finite'),
    ],
)
def test_argument_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        modelfree.guinier(sphere(), **kwargs)


def test_volume_fraction_validation():
    with pytest.raises(ValueError, match='between 0 and 1'):
        modelfree.invariant(sphere(), volume_fraction=1.0)


def test_warnings_are_emitted_too():
    with pytest.warns(UserWarning, match=r'\[qr_limit\]'):
        warnings.simplefilter('always')
        modelfree.guinier(sphere(), qmax=0.05)


# =============================================================================
# Results, reports and export
# =============================================================================


@pytest.fixture(scope='module')
def chain():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = polydisperse_spheres()
        p = modelfree.porod(data, qmin=0.15)
        g = modelfree.guinier(data, background=p.background)
        return {
            'data': data,
            'porod': p,
            'guinier': g,
            'power_law': modelfree.power_law(data, qmin=0.15, qmax=0.6, fit_background=True),
            'kratky': modelfree.kratky(data, guinier=g),
            'invariant': modelfree.invariant(data, low_q=g, high_q=p, delta_sld=3),
        }


@pytest.mark.parametrize('name', ['guinier', 'porod', 'power_law'])
def test_fit_report_renders(chain, name):
    result = chain[name]
    report = result.report()
    assert report.engine == 'modelfree'
    assert set(report.cov_labels) <= set(report.parameters)
    assert report.notes and report.notes[0].startswith('Q window')
    for text in (str(report), report.to_markdown(), report._repr_html_()):
        assert report.notes[0].split()[0] in text
    payload = report.to_dict()
    assert payload['notes'] == report.notes
    json.dumps(payload, allow_nan=False)
    assert report.chisq == pytest.approx(
        np.sum(((result.intensity - result.fit) / result.sigma) ** 2)
    )


def test_guinier_report_parameters(chain):
    report = chain['guinier'].report()
    assert list(report.parameters) == ['rg', 'i0', 'background']
    assert report.parameters['background']['fixed']
    assert report.cov_labels == ['rg', 'i0']


@pytest.mark.parametrize('name', ['guinier', 'porod', 'power_law', 'kratky', 'invariant'])
def test_summary_dict_csv_and_plot(chain, name, tmp_path):
    result = chain[name]
    assert result.title in result.format_summary()
    json.dumps(result.to_dict(), allow_nan=False)
    path = tmp_path / 'out.csv'
    result.save_csv(str(path))
    rows = [
        line for line in path.read_text(encoding='utf-8').splitlines() if not line.startswith('#')
    ]
    if name != 'invariant':
        assert len(rows) - 1 == result.n_points
    else:
        assert {'measured', 'low_q', 'high_q'} <= {row.split(',')[-1] for row in rows[1:]}
    fig = result.plot(chain['data'], show=False)
    assert len(fig.data) >= 1


def test_csv_header_counts_excluded_points(tmp_path):
    data = sphere()
    data.y[3] = -1.0
    g = modelfree.guinier(data)
    path = tmp_path / 'g.csv'
    g.save_csv(str(path))
    header = [
        line for line in path.read_text(encoding='utf-8').splitlines() if 'I - B <= 0' in line
    ]
    assert header and header[0].split()[-1] == '1'


def test_fit_report_unchanged_without_notes():
    from iquana.report import FitReport

    report = FitReport(
        model='m',
        engine='e',
        method='x',
        resolution='r',
        weighting_note='dI',
        chisq=1.0,
        reduced_chisq=1.0,
        n_points=3,
        n_free=1,
        dof=2,
        converged=True,
        message='',
        parameters={},
    )
    assert 'Notes' not in str(report)
    assert '<ul>' not in report._repr_html_()
    assert report.to_dict()['notes'] == []


def test_starting_values_are_plain_floats(chain):
    g, p = chain['guinier'], chain['porod']
    assert isinstance(g.sphere_radius, float)
    assert isinstance(p.background, float)
