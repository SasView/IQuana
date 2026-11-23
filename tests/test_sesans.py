"""SESANS (and reduced SEMSANS) data: loading, the range, guards, plots, export,
persistence, series and the sensitivity warning. See docs/sesans.md.

The data files are sasdata's own examples, read from the installed package:
``sphere2micron.ses`` (Delft, one wavelength, 2 um polystyrene spheres) and
``sphere_isis.ses`` (ISIS, time of flight).
"""

import contextlib
import copy
import io
import json
import math
import os
import re
import warnings

import numpy as np
import pytest
import sasdata
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from scipy import integrate

from iquana import IQuana, MultiFitter, data_ops
from iquana.data.loader import get_fit_index
from iquana.data.sesans import acceptance, wavelength
from iquana.fitter import TRANSFORM_WARNING
from iquana.fitting.base import select_fit_points
from iquana.persistence import SCHEMA_VERSION, AnalysisFileError, build_analysis_dict

SES_DIR = os.path.join(os.path.dirname(sasdata.__file__), 'example_data', 'sesans_data')
DELFT = os.path.join(SES_DIR, 'sphere2micron.ses')
ISIS = os.path.join(SES_DIR, 'sphere_isis.ses')


def quiet(function, *args, **kwargs):
    """Call *function* with its console output and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def sphere_fitter(source=DELFT, radius=9000.0):
    """A sphere fitter on *source* with radius and scale free, contrast fixed at 1."""
    fitter = IQuana()
    if isinstance(source, str):
        quiet(fitter.load_data, source)
    else:
        quiet(fitter.set_data, source)
    quiet(fitter.set_model, 'sphere')
    quiet(fitter.set_param, 'radius', value=radius, min=500, max=30000, vary=True)
    quiet(fitter.set_param, 'scale', value=0.01, min=1e-4, max=1, vary=True)
    quiet(fitter.set_param, 'sld', value=1.0, vary=False)
    quiet(fitter.set_param, 'sld_solvent', value=0.0, vary=False)
    return fitter


def edited_ses(path, tmp_path, edit):
    """A copy of the .ses file at *path* with its lines passed through *edit*."""
    lines = open(path, encoding='utf-8').read().splitlines()
    target = os.path.join(str(tmp_path), 'edited.ses')
    with open(target, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(edit(lines)) + '\n')
    return target


def reversed_rows(lines):
    start = next(i for i, line in enumerate(lines) if line.strip() == 'BEGIN_DATA')
    rows = [line for line in lines[start + 2 :] if line.strip()]
    return lines[: start + 2] + rows[::-1]


# =========================================================================
# The sphere reference
# =========================================================================


def projected_sphere(u):
    """G(xi/R) of a sphere by projecting its autocorrelation (Andersson 2008, Eq. 60)."""
    if u >= 2:
        return 0.0

    def gamma(v):
        s = math.hypot(u, v)
        return 1 - 0.75 * s + s**3 / 16

    return 4 / 3 * integrate.quad(gamma, 0, math.sqrt(4 - u * u))[0]


def closed_sphere(u):
    """The closed form, with the square of Andersson Eq. 61 and Bakker Eq. 25 corrected."""
    u = np.clip(np.asarray(u, dtype=float), 1e-12, None)
    root = np.sqrt(np.clip(1 - u**2 / 4, 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        value = root * (1 + u**2 / 8) + u**2 / 2 * (1 - u**2 / 16) * np.log((u / 2) / (1 + root))
    return np.where(u >= 2, 0.0, value)


@pytest.mark.parametrize('u', [0.1, 0.5, 1.0, 1.5, 1.9])
def test_the_corrected_closed_form_matches_the_projection(u):
    assert closed_sphere(u) == pytest.approx(projected_sphere(u), rel=1e-8)


def test_the_printed_closed_form_is_wrong():
    printed = math.sqrt(1 - 0.25) * (1 + 1 / 8) + 0.5 * (1 - 1 / 4) ** 2 * math.log(
        0.5 / (1 + math.sqrt(0.75))
    )
    assert projected_sphere(1.0) == pytest.approx(0.3569545651, rel=1e-9)
    assert printed == pytest.approx(0.6038841707, rel=1e-9)


def test_a_dilute_sphere_matches_the_reference_on_the_delft_grid():
    fitter = sphere_fitter()
    radius, phi = 10000.0, 0.13
    quiet(fitter.set_param, 'radius', value=radius)
    quiet(fitter.set_param, 'scale', value=phi)
    curve = fitter.calculate()
    x = fitter.data.x
    # (3/2) phi drho^2 R, with drho^2 = 1e-12 A^-4 and 1e8 A per cm.
    reference = (
        1.5 * phi * 1e-12 * radius * 1e8 * (np.array([projected_sphere(v) for v in x / radius]) - 1)
    )
    depth = np.abs(reference).max()
    assert np.abs(curve - reference).max() / depth < 1e-3  # measured: 0.05 %


# =========================================================================
# The range adapter
# =========================================================================

PARS = {'radius': 1000.0, 'scale': 0.1, 'background': 0.0, 'sld': 1.0, 'sld_solvent': 0.0}


def _loaded(path):
    return quiet(data_ops.load, path)


@pytest.mark.parametrize(
    'path, select',
    [
        (DELFT, lambda d: setattr(d, 'qmin', 5000) or setattr(d, 'qmax', 30000)),
        (DELFT, lambda d: d.mask.__setitem__(7, True)),
        (DELFT, lambda d: d.y.__setitem__(3, np.nan)),
        (ISIS, lambda d: setattr(d, 'qmax', d.x[-2])),  # drop the longest wavelength
        (ISIS, lambda d: setattr(d, 'qmin', d.x[10]) or setattr(d, 'qmax', d.x[10])),
    ],
    ids=['range', 'mask', 'nan', 'tof-endpoint', 'single-point'],
)
def test_the_selected_curve_is_the_full_grid_curve_at_the_kept_points(path, select):
    kernel = load_model('sphere')
    full_data = _loaded(path)
    full = DirectModel(copy.deepcopy(full_data), kernel)(**PARS)
    data = copy.deepcopy(full_data)
    select(data)
    keep = get_fit_index(data)
    calculator = select_fit_points(DirectModel(data, kernel), data)
    assert np.array_equal(calculator.index, keep)
    assert np.array_equal(calculator(**PARS), full[keep])


def test_the_adapter_leaves_iq_data_alone():
    data = quiet(
        data_ops.load, os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    )
    calculator = DirectModel(data, load_model('sphere'))
    resolution = calculator.resolution
    assert select_fit_points(calculator, data).resolution is resolution


@pytest.mark.parametrize('engine', ['bumps', 'lmfit'])
def test_every_engine_fits_only_the_selected_points(engine):
    fitter = sphere_fitter()
    quiet(fitter.set_q_range, 5000, 30000)
    keep = fitter._fit_index()
    # A wild value at an excluded point must not reach the objective.
    fitter.data.y[np.flatnonzero(~keep)[0]] = 1e3
    result = quiet(fitter.fit, engine)
    assert result['n_points'] == int(keep.sum()) == 22

    curve = fitter.calculate()
    y, dy = fitter.data.y[keep], fitter.data.dy[keep]
    assert result['chisq'] == pytest.approx(np.sum(((y - curve[keep]) / dy) ** 2), rel=1e-5)
    assert result['dof'] == 22 - 2


def test_the_bayesian_fit_honours_the_range():
    fitter = sphere_fitter(radius=10000)
    quiet(fitter.set_q_range, 5000, 30000)
    result = quiet(fitter.fit_bayesian, samples=300, burn=10, thin=1, pop=4)
    assert result['n_points'] == 22
    figure = fitter.plot_posterior_predictive(show=False)
    assert figure.layout.xaxis.type == 'linear'
    assert len(figure.data[-1].x) == 22


# =========================================================================
# Loading and the array constructor
# =========================================================================


def test_a_ses_file_loads_in_canonical_form():
    fitter = sphere_fitter()
    assert fitter.is_sesans and not fitter.is_2d
    assert fitter.data.source.wavelength_unit == 'A'
    assert wavelength(fitter.data).shape == fitter.data.x.shape
    assert acceptance(fitter.data) == pytest.approx(0.0168)


def test_a_file_with_rows_out_of_order_is_refused(tmp_path):
    with pytest.raises(ValueError, match='not in increasing spin-echo length'):
        quiet(IQuana().load_data, edited_ses(ISIS, tmp_path, reversed_rows))


def test_an_acceptance_in_degrees_is_converted(tmp_path):
    def degrees(lines):
        swap = {
            'Theta_zmax ': f'Theta_zmax              {math.degrees(0.0168)!r}',
            'Theta_zmax_unit': 'Theta_zmax_unit         degrees',
        }
        return [next((v for k, v in swap.items() if line.startswith(k)), line) for line in lines]

    fitter = IQuana()
    quiet(fitter.load_data, edited_ses(DELFT, tmp_path, degrees))
    assert acceptance(fitter.data) == pytest.approx(0.0168)


def test_an_unknown_unit_is_refused():
    data = copy.deepcopy(_loaded(DELFT))
    data._xunit = 'parsec'
    with pytest.raises(ValueError, match="Unknown unit 'parsec'"):
        quiet(IQuana().set_data, data)


def test_set_data_never_changes_the_object_it_is_given():
    data = copy.deepcopy(_loaded(DELFT))
    data.source.wavelength = 2.11  # a scalar: the fitter's copy gets one per point
    source = data.source
    fitter = IQuana()
    quiet(fitter.set_data, data)
    assert data.source is source and data.source.wavelength == 2.11
    assert wavelength(fitter.data).shape == data.x.shape


def test_polarisation_is_converted_as_the_file_does():
    # First row of sphere2micron.ses: P = 1.0037 +- 0.0032974, t = 0.2 cm,
    # lambda = 2.11 A; the file's own Depolarisation is 4.1929e-3 +- 3.6894e-3.
    data = data_ops.sesans(
        [391.56, 1564.0],
        polarisation=[1.0037, 0.99586],
        error=[0.0032974, 0.003386],
        wavelength=2.11,
        thickness=0.2,
        acceptance=0.0168,
    )
    assert data.y[0] == pytest.approx(math.log(1.0037) / (0.2 * 2.11**2))
    assert data.y[0] == pytest.approx(4.1929e-3, rel=0.02)
    assert data.dy[0] == pytest.approx(3.6894e-3, rel=0.01)
    assert data.y[0] > 0  # a ratio above one is accepted
    assert wavelength(data).tolist() == [2.11, 2.11]


def test_depolarisation_is_taken_as_it_is():
    loaded = _loaded(DELFT)
    data = data_ops.sesans(
        loaded.x, depolarisation=loaded.y, error=loaded.dy, wavelength=2.11, acceptance=0.0168
    )
    assert np.array_equal(data.y, loaded.y)
    fitter = IQuana()
    assert not fitter.is_sesans
    quiet(fitter.set_data, data)
    assert fitter.is_sesans


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'polarisation': [0.9, 0.0], 'thickness': 0.2}, '1 of 2 polarisation ratios'),
        ({'polarisation': [0.9, 0.8]}, 'needs the sample thickness'),
        ({'depolarisation': [0.0, -1e-3], 'thickness': 0.2}, 'thickness= is used only'),
        ({'depolarisation': [0.0, -1e-3], 'polarisation': [1, 1]}, 'exactly one'),
        ({}, 'exactly one'),
        ({'depolarisation': [0.0]}, 'has 1 values for 2'),
    ],
)
def test_the_constructor_refuses_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        data_ops.sesans([400.0, 800.0], wavelength=2.11, acceptance=0.0168, **kwargs)


@pytest.mark.parametrize('acceptance_angle', [0.0, -0.1, float('nan')])
def test_the_constructor_refuses_a_bad_acceptance(acceptance_angle):
    with pytest.raises(ValueError, match='acceptance angle'):
        data_ops.sesans(
            [400.0, 800.0], depolarisation=[0, -1e-3], wavelength=2.11, acceptance=acceptance_angle
        )


def test_the_constructor_refuses_decreasing_lengths():
    with pytest.raises(ValueError, match='strictly increasing'):
        data_ops.sesans([800.0, 400.0], depolarisation=[0, -1e-3], wavelength=2.11, acceptance=0.01)


# =========================================================================
# Guards
# =========================================================================


def test_resolution_is_refused_by_the_setter():
    fitter = sphere_fitter()
    with pytest.raises(ValueError, match='Resolution smearing does not apply'):
        fitter.set_resolution('pinhole', dq_over_q=0.1)
    assert fitter.get_resolution()['mode'] == 'data'


def test_sesans_data_is_refused_when_a_resolution_is_already_set():
    fitter = IQuana()
    quiet(fitter.set_resolution, 'none')
    with pytest.raises(ValueError, match='Resolution smearing does not apply'):
        quiet(fitter.load_data, DELFT)
    assert fitter.data is None
    with pytest.raises(ValueError, match='Resolution smearing does not apply'):
        quiet(fitter.set_data, _loaded(DELFT))
    assert fitter.data is None


def test_weighting_is_refused_by_the_setter():
    fitter = sphere_fitter()
    with pytest.raises(ValueError, match='SESANS data is always weighted'):
        fitter.set_weighting('sqrt')
    assert fitter.get_weighting()['mode'] == 'dI'


def test_a_free_background_is_refused():
    fitter = sphere_fitter()
    quiet(fitter.set_param, 'background', value=0.0, min=-1, max=1, vary=True)
    with pytest.raises(ValueError, match="'background' is not used for SESANS"):
        quiet(fitter.fit)


def test_missing_errors_name_a_usable_exit():
    loaded = _loaded(DELFT)
    data = data_ops.sesans(loaded.x, depolarisation=loaded.y, wavelength=2.11, acceptance=0.0168)
    fitter = sphere_fitter(data)
    with pytest.raises(ValueError, match='error= of data_ops.sesans') as caught:
        quiet(fitter.fit)
    assert 'set_weighting' not in str(caught.value)


def test_a_custom_grid_is_refused():
    fitter = sphere_fitter()
    with pytest.raises(ValueError, match='own spin-echo lengths'):
        fitter.calculate(q=np.linspace(0.001, 0.01, 5))
    with pytest.raises(ValueError, match='own spin-echo lengths'):
        fitter.compare(radius=[5000, 9000], q=np.linspace(0.001, 0.01, 5), show=False)


def test_multifitter_refuses_both_routes():
    with pytest.raises(ValueError, match='not supported by MultiFitter'):
        quiet(MultiFitter().add, 'a', DELFT, model='sphere')
    with pytest.raises(ValueError, match='not supported by MultiFitter'):
        quiet(MultiFitter().add_fitter, 'a', sphere_fitter())


def test_arithmetic_is_refused():
    with pytest.raises(TypeError, match='SESANS data'):
        data_ops.subtract(_loaded(DELFT), 0.01)


# =========================================================================
# Plots, export, report
# =========================================================================


@pytest.fixture(scope='module')
def fitted():
    fitter = sphere_fitter()
    quiet(fitter.fit)
    return fitter


def test_the_example_fit_recovers_the_spheres(fitted):
    result = fitted.fit_result
    assert result['parameters']['radius']['value'] == pytest.approx(10382, rel=0.02)
    assert result['reduced_chisq'] == pytest.approx(1.05, abs=0.05)


def test_plots_are_linear_and_labelled(fitted):
    figure = fitted.plot_results(show=False)
    assert figure.layout.xaxis2.title.text == 'Spin-echo length (Å)'
    assert figure.layout.yaxis.title.text.startswith('ln(P/P₀)')
    assert figure.layout.yaxis.type == 'linear'
    assert figure.layout.title.text.startswith('SESANS Fit')
    assert fitted.plot_model(show=False).layout.yaxis.type == 'linear'
    assert fitted.compare(radius=[9000, 11000], show=False).layout.xaxis.type == 'linear'
    with pytest.raises(ValueError, match='linear axes'):
        fitted.plot_results(log_scale=True, show=False)


def test_iq_plots_keep_log_axes():
    fitter = IQuana()
    quiet(
        fitter.load_data, os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    )
    quiet(fitter.set_model, 'sphere')
    assert fitter.plot_model(show=False).layout.yaxis.type == 'log'


def test_the_csv_has_sesans_columns(fitted, tmp_path):
    path = os.path.join(str(tmp_path), 'fit.csv')
    quiet(fitted.save_results, path)
    lines = open(path, encoding='utf-8').read().splitlines()
    assert lines[0] == '# SESANS Fit Results'
    header = next(line for line in lines if not line.startswith('#'))
    assert header == 'spin_echo_length,depolarisation,error,fit,residual'


def test_no_user_facing_text_talks_about_q(tmp_path):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitter = sphere_fitter()
        fitter.load_data(DELFT)
        fitter.set_q_range(5000, 30000)
        fitter.reset_q_range()
        fitter.fit()
        report = fitter.report()
    text = output.getvalue() + str(report)
    figure = fitter.plot_results(show=False)
    text += ' '.join(str(axis.title.text) for axis in (figure.layout.xaxis2, figure.layout.yaxis))
    assert not re.search(r'\bQ\b|I\(Q\)|1/Å|Å⁻¹|1/Ang|1/A\b', text), text


# =========================================================================
# Sensitivity warning
# =========================================================================


def _exact_data(radius):
    """sphere2micron's grid with the model itself as data and wide error bars."""
    loaded = _loaded(DELFT)
    fitter = sphere_fitter()
    quiet(fitter.set_param, 'radius', value=radius)
    quiet(fitter.set_param, 'scale', value=0.13)
    curve = fitter.calculate()
    return data_ops.sesans(
        loaded.x,
        depolarisation=curve,
        error=np.ones_like(curve),
        wavelength=2.11,
        acceptance=0.0168,
    )


@pytest.mark.parametrize('radius, warns', [(1000.0, True), (2000.0, False), (10000.0, False)])
def test_the_sensitivity_warning_tracks_the_transform_error(radius, warns):
    fitter = sphere_fitter(_exact_data(radius), radius=radius)
    quiet(fitter.set_param, 'scale', value=0.13)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitter._warn_if_transform_sensitive()
    texts = [str(w.message) for w in caught if str(w.message).startswith(TRANSFORM_WARNING)]
    assert bool(texts) is warns


def test_the_fit_warns_and_still_returns_its_result():
    fitter = sphere_fitter(_exact_data(1000.0), radius=1200.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = quiet(fitter.fit)
        fitter.fit()
    assert result['parameters']['radius']['value'] > 0
    assert any(str(w.message).startswith(TRANSFORM_WARNING) for w in caught)


# =========================================================================
# Persistence
# =========================================================================


def round_trip(fitter, tmp_path, **kwargs):
    path = os.path.join(str(tmp_path), 'analysis.json')
    quiet(fitter.save_analysis, path)
    return quiet(IQuana.load_analysis, path, **kwargs), path


def test_a_file_analysis_round_trips_with_its_result(tmp_path):
    fitter = sphere_fitter()
    quiet(fitter.set_q_range, 5000, 30000)
    quiet(fitter.fit)
    loaded, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['schema']['version'] == SCHEMA_VERSION == 4
    assert document['data']['measurement'] == 'sesans'
    assert loaded.is_sesans and loaded.fit_result is not None
    assert loaded.fit_result['n_points'] == 22
    assert loaded.get_q_range() == pytest.approx((5000, 30000))


def test_an_in_memory_analysis_round_trips_with_its_result(tmp_path):
    loaded_file = _loaded(ISIS)
    data = data_ops.sesans(
        loaded_file.x,
        depolarisation=loaded_file.y,
        error=loaded_file.dy,
        wavelength=wavelength(loaded_file),
        acceptance=0.09,
    )
    fitter = sphere_fitter(data, radius=1000)
    quiet(fitter.fit)
    loaded, path = round_trip(fitter, tmp_path)
    assert loaded.fit_result is not None
    assert np.array_equal(wavelength(loaded.data), wavelength(fitter.data))
    assert acceptance(loaded.data) == 0.09


def test_a_changed_wavelength_drops_the_result(tmp_path):
    fitter = sphere_fitter()
    quiet(fitter.fit)
    path = os.path.join(str(tmp_path), 'analysis.json')
    quiet(fitter.save_analysis, path)
    other = copy.deepcopy(fitter.data)
    other.source.wavelength = wavelength(other) * 2
    loaded = quiet(IQuana.load_analysis, path, data=other)
    assert loaded.fit_result is None


def test_data_of_the_other_kind_is_refused(tmp_path):
    fitter = sphere_fitter()
    quiet(fitter.fit)
    _, path = round_trip(fitter, tmp_path)
    iq = os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    with pytest.raises(AnalysisFileError, match='saved with SESANS data'):
        quiet(IQuana.load_analysis, path, data=iq)


def test_an_older_sesans_analysis_loads_without_its_result(tmp_path):
    fitter = sphere_fitter()
    quiet(fitter.fit)
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    document['schema']['version'] = 3
    del document['data']['measurement']
    json.dump(document, open(path, 'w', encoding='utf-8'))
    loaded = quiet(IQuana.load_analysis, path)
    assert loaded.is_sesans and loaded.fit_result is None


def test_an_iq_analysis_is_still_written_as_version_3(tmp_path):
    fitter = IQuana()
    quiet(
        fitter.load_data, os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    )
    quiet(fitter.set_model, 'sphere')
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['schema']['version'] == 3
    assert 'measurement' not in document['data']


# =========================================================================
# Series
# =========================================================================


def _rows():
    """Three in-memory SESANS datasets, the Delft curve scaled, with distinct titles."""
    loaded = _loaded(DELFT)
    rows = []
    for factor in (0.8, 1.0, 1.2):
        data = data_ops.sesans(
            loaded.x,
            depolarisation=loaded.y * factor,
            error=loaded.dy,
            wavelength=2.11,
            acceptance=0.0168,
        )
        data.title = f'scaled {factor}'
        rows.append(data)
    return rows


def test_a_sesans_series_fits_replays_the_range_and_reloads(tmp_path):
    template = sphere_fitter(radius=10000)
    quiet(template.set_q_range, 5000, 30000)
    series = quiet(template.fit_series, _rows(), values=[1, 2, 3])
    assert series.is_sesans
    assert [row.result['n_points'] for row in series] == [22, 22, 22]
    path = os.path.join(str(tmp_path), 'series.json')
    quiet(series.save_analysis, path)
    reloaded = quiet(type(series).load_analysis, path)
    assert reloaded.is_sesans and len(reloaded) == 3
    assert reloaded.plot_fits(show=False).layout.xaxis.type == 'linear'


def test_a_template_without_data_takes_the_kind_of_the_first_dataset():
    template = IQuana()
    quiet(template.set_model, 'sphere')
    quiet(template.set_param, 'radius', value=10000, min=500, max=30000, vary=True)
    quiet(template.set_param, 'scale', value=0.01, min=1e-4, max=1, vary=True)
    quiet(template.set_param, 'sld', value=1.0, vary=False)
    quiet(template.set_param, 'sld_solvent', value=0.0, vary=False)
    iq = os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    series = quiet(template.fit_series, [*_rows()[:2], iq])
    assert series.is_sesans
    assert not series[0].failed and not series[1].failed
    assert series[2].failed and 'series holds SESANS data' in series[2].error


def test_a_free_background_stops_a_series_without_data_at_the_first_dataset():
    template = IQuana()
    quiet(template.set_model, 'sphere')
    quiet(template.set_param, 'background', value=0.0, min=-1, max=1, vary=True)
    with pytest.raises(ValueError, match="'background' is not used"):
        quiet(template.fit_series, _rows())


def test_update_refuses_a_fitter_of_the_other_kind():
    template = sphere_fitter(radius=10000)
    series = quiet(template.fit_series, _rows(), values=[1, 2, 3])
    other = IQuana()
    quiet(
        other.load_data, os.path.join(os.path.dirname(__file__), 'data', 'simulated_sans_data.csv')
    )
    quiet(other.set_model, 'sphere')
    quiet(other.set_param, 'radius', vary=True)
    quiet(other.fit)
    with pytest.raises(ValueError, match='the series holds SESANS data'):
        series._check_same_kind(build_analysis_dict(other, None))
