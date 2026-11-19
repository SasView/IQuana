"""Series (batch) fitting: IQuana.fit_series() and SeriesResult.

Simulated scans with a known parameter ramp, written to files the way a
reduction pipeline delivers them. Sections follow the design document:
the fit, items/values/order, the Q cut, failures and the chain, and the
result (repair, persistence).
"""

import json
import logging
import os
import shutil
import signal
import warnings

import numpy as np
import pytest

import iquana.series as series_module
import iquana.series_results as results_module
from iquana import IQuana, SeriesResult, examples, set_verbosity
from iquana.console import logger
from iquana.persistence import AnalysisFileError
from iquana.series_results import SeriesRow, _assess

TEMPERATURES = [20, 22, 24, 26, 28, 30]


def write_dataset(data, path):
    np.savetxt(path, np.column_stack([data.x, data.y, data.dy]))


def simulate(radius, seed=0, qmax=0.4, npoints=60):
    return examples.simulate(
        'sphere',
        radius=radius,
        scale=0.01,
        background=0.001,
        noise=0.02,
        seed=seed,
        qmin=0.005,
        qmax=qmax,
        npoints=npoints,
    )


def write_scan(directory, temperatures=TEMPERATURES, radius=lambda i: 40.0 + i):
    """T_<t>K.dat files, the radius rising by 1 Å per file."""
    os.makedirs(directory, exist_ok=True)
    for i, t in enumerate(temperatures):
        write_dataset(simulate(radius(i), seed=i), os.path.join(directory, f'T_{t:03d}K.dat'))
    return directory


def make_template(data_path=None):
    f = IQuana()
    if data_path is not None:
        f.load_data(data_path)
    f.set_model('sphere')
    f.set_param('radius', value=40, min=10, max=100, vary=True)
    f.set_param('scale', value=0.01, min=1e-4, max=1, vary=True)
    f.set_param('background', value=0.001, min=0, max=1, vary=True)
    f.set_resolution('none')
    return f


@pytest.fixture
def scan(tmp_path):
    return write_scan(str(tmp_path / 'scan'))


@pytest.fixture
def pattern(scan):
    return os.path.join(scan, 'T_{temperature}K.dat')


@pytest.fixture
def no_fit(monkeypatch):
    """Fail the test if any fit runs: the call must be refused up front."""

    def boom(*_args, **_kwargs):
        raise AssertionError('a fit ran although the call should have been refused')

    monkeypatch.setattr(IQuana, 'fit', boom)


# =============================================================================
# The fit
# =============================================================================


@pytest.mark.parametrize('engine', ['bumps', 'lmfit'])
@pytest.mark.parametrize('start', ['previous', 'template'])
def test_truth_is_recovered_for_both_start_modes_and_engines(pattern, engine, start):
    series = make_template().fit_series(pattern, start=start, engine=engine)
    assert len(series) == len(TEMPERATURES)
    for i, row in enumerate(series):
        value, err = row.estimate('radius')
        assert value == pytest.approx(40.0 + i, abs=max(5 * err, 0.5))
    assert series._counts() == {'ok': len(TEMPERATURES), 'flagged': 0, 'failed': 0}


def test_a_series_of_one_file_equals_a_single_fit(scan):
    path = os.path.join(scan, 'T_024K.dat')
    series = make_template().fit_series(path, engine='lmfit')
    single = make_template(path)
    single.fit(engine='lmfit')
    for name in ('radius', 'scale', 'background'):
        assert series[0].estimate(name)[0] == pytest.approx(single.params[name]['value'], rel=1e-12)
    assert series[0].chisq == pytest.approx(single.fit_result['chisq'], rel=1e-12)


def test_the_template_is_not_modified(scan, pattern):
    template = make_template(os.path.join(scan, 'T_020K.dat'))
    template.set_q_range(qmin=0.01)
    template.fit()
    before = (
        template.data,
        template.fit_result,
        template.get_q_range(),
        json.dumps(template.params),
    )
    template.fit_series(pattern)
    after = (
        template.data,
        template.fit_result,
        template.get_q_range(),
        json.dumps(template.params),
    )
    assert before[0] is after[0] and before[1] is after[1]
    assert before[2:] == after[2:]


def test_the_setup_reaches_every_row(pattern):
    template = make_template()
    template.set_resolution('pinhole', dq_over_q=0.05)
    template.set_weighting('sqrt')
    template.set_pd_param('radius', pd_width=0.05, pd_n=15, vary=False)
    template.enable_polydispersity(True)
    series = template.fit_series(pattern)
    for row in series:
        assert not row.failed
        assert row.analysis['resolution']['mode'] == 'pinhole'
        assert row.analysis['weighting']['mode'] == 'sqrt'
        assert row.analysis['configuration']['polydispersity']['enabled'] is True


def test_a_structure_factor_and_a_composite_model_reach_every_row(pattern):
    template = make_template()
    template.set_structure_factor('hardsphere')
    series = template.fit_series(pattern)
    assert all(
        r.analysis['configuration']['structure_factor']['name'] == 'hardsphere' for r in series
    )

    composite = IQuana()
    composite.set_models('sphere', 'power_law')
    composite.set_resolution('none')
    composite.set_param('sphere_radius', value=40, min=10, max=100, vary=True)
    composite.set_param('scale', value=0.01, min=1e-4, max=1, vary=True)
    series = composite.fit_series(pattern)
    assert all(r.analysis['configuration']['components'] for r in series)


def test_an_all_fixed_template_is_refused_by_fit_and_fit_series(scan, pattern, no_fit):
    template = make_template()
    for name in ('radius', 'scale', 'background'):
        template.set_param(name, vary=False)
    with pytest.raises(ValueError, match='No parameter is free'):
        template.fit_series(pattern)


def test_fit_refuses_an_all_fixed_model_and_accepts_a_pd_width(scan):
    single = make_template(os.path.join(scan, 'T_020K.dat'))
    for name in ('radius', 'scale', 'background'):
        single.set_param(name, vary=False)
    with pytest.raises(ValueError, match='No parameter is free'):
        single.fit()
    single.set_pd_param('radius', pd_width=0.05, vary=True)
    single.enable_polydispersity(True)
    single.fit()  # a polydispersity width alone is something to fit


def test_differential_evolution_notes_once_that_it_ignores_the_chain(pattern, monkeypatch, capsys):
    monkeypatch.setattr(IQuana, 'fit', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x')))
    make_template().fit_series(pattern, engine='lmfit', method='differential_evolution')
    assert (
        capsys.readouterr().out.count("method='differential_evolution' starts from the bounds") == 1
    )


def test_the_report_is_not_built_when_it_will_not_be_shown(scan, pattern, monkeypatch):
    calls = []
    original = IQuana.get_fit_report
    monkeypatch.setattr(IQuana, 'get_fit_report', lambda self: calls.append(1) or original(self))
    make_template().fit_series(pattern)
    assert calls == []
    single = make_template(os.path.join(scan, 'T_020K.dat'))
    set_verbosity('quiet')
    try:
        single.fit()
    finally:
        set_verbosity('info')
    assert calls == []


# =============================================================================
# Items, values, order
# =============================================================================


def test_placeholder_values_are_read_and_sorted(pattern):
    series = make_template().fit_series(pattern)
    assert series.variable == 'temperature'
    assert [row.value for row in series] == [float(t) for t in TEMPERATURES]
    assert [row.label for row in series] == [f'T_{t:03d}K.dat' for t in TEMPERATURES]


def test_natural_order_list_order_and_literal_glob_characters(tmp_path, scan):
    for name in ('run10.dat', 'run2.dat', 'run1.dat'):
        shutil.copy(os.path.join(scan, 'T_020K.dat'), tmp_path / name)
    series = make_template().fit_series(str(tmp_path / 'run*.dat'))
    assert [row.label for row in series] == ['run1.dat', 'run2.dat', 'run10.dat']
    assert series.variable == 'index'

    files = [str(tmp_path / 'run10.dat'), str(tmp_path / 'run1.dat')]
    assert [row.label for row in make_template().fit_series(files)] == ['run10.dat', 'run1.dat']

    shutil.copy(os.path.join(scan, 'T_020K.dat'), tmp_path / 'a[1]_x.dat')
    shutil.copy(os.path.join(scan, 'T_020K.dat'), tmp_path / 'a1_x.dat')
    literal = make_template().fit_series(str(tmp_path / 'a[1]_*.dat'))
    assert [row.label for row in literal] == ['a[1]_x.dat']


def test_a_single_path_gives_one_row(scan):
    series = make_template().fit_series(os.path.join(scan, 'T_026K.dat'))
    assert len(series) == 1 and series[0].label == 'T_026K.dat'


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda t, p, s: t.fit_series(os.path.join(s, 'X_*.dat')), 'No files match'),
        (lambda t, p, s: t.fit_series(os.path.join(s, '{a}_{b}.dat')), 'placeholders'),
        (lambda t, p, s: t.fit_series(os.path.join(s, '{t}.dat')), 'has no number'),
        (lambda t, p, s: t.fit_series([os.path.join(s, 'T_020K.dat')], values=[1, 2]), 'entries'),
        (lambda t, p, s: t.fit_series(p, values=[1, 2, 3, 4, 5, 6]), 'list of datasets'),
        (lambda t, p, s: t.fit_series(p, values='sample.temperature'), 'Two sources'),
        (lambda t, p, s: t.fit_series([os.path.join(s, 'T_020K.dat')] * 2), 'unique'),
        (lambda t, p, s: t.fit_series(p, variable='Temperature (K)'), 'identifier'),
        (lambda t, p, s: t.fit_series(p, variable='radius_err'), 'collides'),
        (lambda t, p, s: t.fit_series(p, variable='chisq'), 'collides'),
        (lambda t, p, s: t.fit_series(p, start='middle'), 'Unknown start'),
        (lambda t, p, s: t.fit_series(p, engine='nope'), 'Unknown engine'),
        (lambda t, p, s: t.fit_series(os.path.join(s, 'missing.dat')), 'No such file'),
    ],
)
def test_mistakes_in_the_call_are_refused_before_any_fit(pattern, scan, no_fit, call, message):
    with pytest.raises(ValueError, match=message):
        call(make_template(), pattern, scan)


def test_a_template_without_a_model_is_refused(pattern, no_fit):
    with pytest.raises(ValueError, match='No model set'):
        IQuana().fit_series(pattern)


def test_name_order_keeps_file_order_with_repeated_values(tmp_path):
    fields = [0.5, 1.0, 0.5, 0.0]  # up and back down: a loop
    for i, field in enumerate(fields):
        write_dataset(simulate(40.0, seed=i), tmp_path / f'run{i + 1}_H{field}T.dat')
    series = make_template().fit_series(str(tmp_path / 'run*_H{field}T.dat'), order='name')
    assert [row.value for row in series] == fields
    by_value = make_template().fit_series(str(tmp_path / 'run*_H{field}T.dat'))
    assert [row.value for row in by_value] == sorted(fields)


def test_order_with_a_list_and_value_order_without_a_value_are_refused(scan, no_fit):
    with pytest.raises(ValueError, match='order= applies to a pattern'):
        make_template().fit_series([os.path.join(scan, 'T_020K.dat')], order='name')
    with pytest.raises(ValueError, match="order='value' needs a series value"):
        make_template().fit_series(os.path.join(scan, 'T_*K.dat'), order='value')


def _xml_with_temperature(path, temperature, unit='K'):
    data = simulate(40.0)
    rows = '\n'.join(
        f'<Idata><Q unit="1/A">{q:.6g}</Q><I unit="1/cm">{i:.6g}</I><Idev unit="1/cm">{e:.6g}</Idev></Idata>'
        for q, i, e in zip(data.x, data.y, data.dy, strict=True)
    )
    unit_attr = f' unit="{unit}"' if unit else ''
    path.write_text(
        '<?xml version="1.0"?>\n<SASroot version="1.0" xmlns="cansas1d/1.0">\n'
        f'<SASentry><Title>{path.stem}</Title><SASdata>{rows}</SASdata>'
        f'<SASsample><ID>s</ID><thickness unit="mm">1.0</thickness>'
        f'<temperature{unit_attr}>{temperature}</temperature></SASsample></SASentry></SASroot>'
    )


def test_metadata_values_are_read_and_sorted_by(tmp_path):
    for name, temperature in (('b', 300), ('a', 310), ('c', 290)):
        _xml_with_temperature(tmp_path / f'{name}.xml', temperature)
    series = make_template().fit_series(str(tmp_path / '*.xml'), values='sample.temperature')
    assert series.variable == 'temperature'
    assert [row.value for row in series] == [290.0, 300.0, 310.0]
    assert [row.label for row in series] == ['c.xml', 'b.xml', 'a.xml']


def test_a_missing_metadata_value_lists_what_is_there(tmp_path, no_fit):
    _xml_with_temperature(tmp_path / 'a.xml', 300)
    with pytest.raises(ValueError, match=r"no value at 'sample\.transmission'") as raised:
        make_template().fit_series(str(tmp_path / '*.xml'), values='sample.transmission')
    assert 'sample.temperature = 300' in str(raised.value)
    assert 'sample.thickness = 1' in str(raised.value)


def test_differing_metadata_units_are_refused(tmp_path, no_fit):
    _xml_with_temperature(tmp_path / 'a.xml', 300, unit='K')
    _xml_with_temperature(tmp_path / 'b.xml', 25, unit='C')
    with pytest.raises(ValueError, match='disagree on the unit'):
        make_template().fit_series(str(tmp_path / '*.xml'), values='sample.temperature')


def test_metadata_values_of_in_memory_datasets(no_fit):
    datasets = [simulate(40.0, seed=i) for i in range(2)]
    for i, d in enumerate(datasets):
        d.title = f'frame{i}'
        d.sample.temperature = 10.0 * (i + 1)
    items, variable = series_module._resolve_items(datasets, 'sample.temperature', None, None)
    assert variable == 'temperature' and [item.value for item in items] == [10.0, 20.0]


# =============================================================================
# Q cut
# =============================================================================


def test_a_one_sided_cut_keeps_the_open_end_of_each_dataset(tmp_path):
    narrow, wide = tmp_path / 'f1.dat', tmp_path / 'f2.dat'
    write_dataset(simulate(40.0, qmax=0.2), narrow)
    write_dataset(simulate(40.0, qmax=0.4, seed=1), wide)
    template = make_template(str(narrow))
    template.set_q_range(qmin=0.03)
    series = template.fit_series([str(narrow), str(wide)])
    assert series[1].analysis['fit_range'] == {'qmin': 0.03, 'qmax': None}
    assert series[1].analysis['result']['fit_context']['qmax'] == pytest.approx(0.4)

    template.set_q_range(qmin=0.03, qmax=0.15)
    held = template.fit_series([str(narrow), str(wide)])
    assert held[1].analysis['result']['fit_context']['qmax'] == pytest.approx(0.15)


def test_a_rejected_cut_keeps_the_previous_request(scan):
    f = make_template(os.path.join(scan, 'T_020K.dat'))
    f.set_q_range(qmin=0.03)
    with pytest.raises(ValueError, match='No data points'):
        f.set_q_range(qmin=5.0, qmax=6.0)
    assert f._q_range_request == (0.03, None)


def test_the_q_request_survives_a_saved_analysis(tmp_path):
    narrow, wide = tmp_path / 'f1.dat', tmp_path / 'f2.dat'
    write_dataset(simulate(40.0, qmax=0.2), narrow)
    write_dataset(simulate(40.0, qmax=0.4, seed=1), wide)
    f = make_template(str(narrow))
    f.set_q_range(qmin=0.03)
    f.save_analysis(str(tmp_path / 'a.json'))
    document = json.loads((tmp_path / 'a.json').read_text())
    assert document['fit_range'] == {'qmin': 0.03, 'qmax': None}
    reloaded = IQuana.load_analysis(str(tmp_path / 'a.json'), data=str(wide))
    assert reloaded.get_q_range()[1] == pytest.approx(0.4)

    # An older file's two numbers are honoured as they were written.
    document['fit_range'] = {'qmin': 0.03, 'qmax': 0.1}
    (tmp_path / 'b.json').write_text(json.dumps(document))
    old = IQuana.load_analysis(str(tmp_path / 'b.json'), data=str(wide))
    assert old.get_q_range() == pytest.approx((0.03, 0.1))


def test_a_dataset_with_no_points_in_the_cut_fails_rather_than_fitting_its_full_range(tmp_path):
    low, high = tmp_path / 'f1.dat', tmp_path / 'f2.dat'
    write_dataset(simulate(40.0, qmax=0.4), low)
    write_dataset(examples.simulate('sphere', radius=40, qmin=0.005, qmax=0.02, npoints=30), high)
    template = make_template(str(low))
    template.set_q_range(qmin=0.1, qmax=0.3)
    series = template.fit_series([str(low), str(high)])
    assert series[1].failed and 'No data points' in series[1].error
    assert series[1].n_points is None
    assert series[1].analysis is not None  # it loaded, so it can be opened


# =============================================================================
# Failures and the chain
# =============================================================================


def test_a_bad_file_fails_alone_and_the_chain_skips_it(scan):
    with open(os.path.join(scan, 'T_025K.dat'), 'w') as handle:
        handle.write('not data\n')
    series = make_template().fit_series(os.path.join(scan, 'T_{temperature}K.dat'))
    bad = series['T_025K.dat']
    assert bad.failed and bad.analysis is None
    after = series[bad.index + 1]
    assert not after.failed
    assert after.start_row == bad.index - 1
    assert series._counts() == {'ok': len(TEMPERATURES), 'flagged': 0, 'failed': 1}


def test_a_leastsq_row_outside_its_bounds_is_flagged_and_not_chained(tmp_path):
    for i, radius in enumerate((40.0, 60.0, 41.0)):
        write_dataset(simulate(radius, seed=i), tmp_path / f'r{i}.dat')
    template = make_template()
    template.set_param('radius', value=40, min=10, max=50)
    series = template.fit_series(str(tmp_path / 'r*.dat'), engine='lmfit', method='leastsq')
    outside = series[1]
    assert 'out_of_bounds' in outside.flag_names
    assert not outside.usable_as_start
    assert series[2].start_row == 0

    path = str(tmp_path / 'series.json')
    series.save_analysis(path)
    reloaded = SeriesResult.load_analysis(path)
    assert 'out_of_bounds' in reloaded[1].flag_names
    for row in reloaded:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert reloaded.fitter(row.index).data is not None


def test_a_missing_file_can_be_retried_once_it_exists(tmp_path, scan):
    missing = str(tmp_path / 'later.dat')
    series = make_template().fit_series([os.path.join(scan, 'T_020K.dat'), missing])
    assert series[1].failed and series[1].analysis is None
    with pytest.raises(ValueError):
        series.fitter(1)
    shutil.copy(os.path.join(scan, 'T_022K.dat'), missing)
    repaired = series.fitter(1)
    repaired.fit()
    series.update(1, repaired)
    assert not series[1].failed and series[1].edited


def _row(index, radius, err=0.1, converged=True, failed=False, on_bounds=()):
    """A synthetic row: enough of an analysis document for the flag rules."""
    if failed:
        return SeriesRow(index, f'r{index}', float(index), None, None, 'it broke')
    result = {
        'parameters': {'radius': {'value': radius, 'stderr': err, 'fixed': False}},
        'converged': converged,
        'message': 'stopped',
        'on_bounds': [list(hit) for hit in on_bounds],
        'reduced_chisq': 1.0,
        'chisq': 50.0,
        'n_points': 50,
    }
    configuration = {'params': {'radius': {'value': radius, 'min': 0, 'max': 100, 'vary': True}}}
    return SeriesRow(
        index,
        f'r{index}',
        float(index),
        None,
        {'result': result, 'configuration': configuration},
        None,
    )


def flags_of(rows):
    return {row.index: row.flag_names for row in _assess(rows, 'index') if row.flags}


def test_a_smooth_ramp_has_no_flags():
    assert flags_of([_row(i, 40.0 + i) for i in range(10)]) == {}


def test_a_discontinuity_is_flagged_where_it_happens():
    radii = [40.0 + i for i in range(10)]
    radii[6:] = [r + 30 for r in radii[6:]]
    assert flags_of([_row(i, r) for i, r in enumerate(radii)]) == {6: ('jump',)}


def test_a_jump_inside_the_uncertainty_is_not_flagged():
    radii = [40.0 + i for i in range(10)]
    radii[6:] = [r + 30 for r in radii[6:]]
    assert flags_of([_row(i, r, err=20.0) for i, r in enumerate(radii)]) == {}


def test_too_few_steps_never_flag_a_jump():
    assert flags_of([_row(i, r) for i, r in enumerate((40, 41, 42, 90, 43))]) == {}


def test_no_step_is_taken_across_a_failed_or_unconverged_row():
    rows = [_row(i, 40.0 + i) for i in range(10)]
    rows[5] = _row(5, 0, failed=True)
    rows[6] = _row(6, 99.0, converged=False)  # no step into it, no step out of it
    rows[7] = _row(7, 47.0)
    assert flags_of(rows) == {5: ('failed',), 6: ('unconverged',)}


def test_a_bound_hit_is_flagged_by_a_real_fit(tmp_path):
    for i in range(3):
        write_dataset(simulate(60.0, seed=i), tmp_path / f'r{i}.dat')
    template = make_template()
    template.set_param('radius', value=40, min=10, max=50)
    series = template.fit_series(str(tmp_path / 'r*.dat'))
    assert all('bound' in row.flag_names for row in series)
    assert 'radius at upper bound (50)' in series[0].why()


def test_an_interrupt_returns_the_finished_rows_and_restores_state(pattern, monkeypatch):
    calls = []
    real_fit = IQuana.fit

    def interrupted_fit(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(IQuana, 'fit', interrupted_fit)
    handler, level = signal.getsignal(signal.SIGINT), logger.level
    with pytest.warns(UserWarning, match='2 of 6 datasets processed'):
        series = make_template().fit_series(pattern)
    assert len(series) == 2 and series.interrupted
    assert signal.getsignal(signal.SIGINT) is handler and logger.level == level


def test_an_interrupt_the_engine_swallows_still_stops_the_run(pattern, monkeypatch):
    calls = []
    real_fit = IQuana.fit

    def swallowing_fit(self, *args, **kwargs):
        calls.append(1)
        result = real_fit(self, *args, **kwargs)
        if len(calls) == 2:
            try:  # what bumps' de and DREAM do: catch the interrupt and return
                signal.raise_signal(signal.SIGINT)
            except KeyboardInterrupt:
                pass
        return result

    monkeypatch.setattr(IQuana, 'fit', swallowing_fit)
    with pytest.warns(UserWarning, match='Series interrupted'):
        series = make_template().fit_series(pattern)
    assert len(series) == 1  # the row cut short is dropped, nothing after it starts


def test_state_is_restored_after_an_exception(pattern, monkeypatch):
    handler, level = signal.getsignal(signal.SIGINT), logger.level
    monkeypatch.setattr(series_module, '_Progress', None)  # fails inside the run
    with pytest.raises(TypeError):
        make_template().fit_series(pattern)
    assert signal.getsignal(signal.SIGINT) is handler and logger.level == level


def test_progress_lines_are_throttled(pattern, monkeypatch, capsys):
    times = iter([0.0, 1.0, 2.0, 15.0, 16.0, 17.0, 30.0, 31.0])
    monkeypatch.setattr(series_module, '_clock', lambda: next(times))
    make_template().fit_series(pattern)
    lines = [line for line in capsys.readouterr().out.splitlines() if 'temperature=' in line]
    # Start at t=0; rows end at 1, 2, 15, 16, 17, 30; the summary reads 31.
    assert [line.split()[0] for line in lines] == ['#0', '#2', '#5']
    assert '[1/6,' in lines[0]


def test_debug_verbosity_prints_every_row(pattern, capsys):
    set_verbosity('debug')
    try:
        make_template().fit_series(pattern)
    finally:
        set_verbosity('info')
    out = capsys.readouterr().out
    assert sum('temperature=' in line for line in out.splitlines()) == len(TEMPERATURES)


# =============================================================================
# Result, repair, persistence
# =============================================================================


@pytest.fixture
def fitted(pattern):
    return make_template().fit_series(pattern)


def test_table_dataframe_csv_and_renderings_agree(fitted, tmp_path):
    pd = pytest.importorskip('pandas')
    table = fitted.table()
    assert list(table[0])[:3] == ['row', 'label', 'temperature']
    assert {'radius', 'radius_err', 'scale', 'background'} <= set(table[0])
    frame = fitted.to_dataframe()
    assert list(frame.index) == [row['row'] for row in table]
    assert frame['radius'].tolist() == pytest.approx([row['radius'] for row in table])

    fitted.save_table(str(tmp_path / 'scan.csv'))
    csv_frame = pd.read_csv(tmp_path / 'scan.csv')
    assert csv_frame['radius'].tolist() == pytest.approx(frame['radius'].tolist())

    text, html = str(fitted), fitted._repr_html_()
    for row in table:
        assert row['label'] in text and row['label'] in html


def test_zero_and_missing_uncertainties_show_as_unavailable():
    rows = [_row(0, 40.0, err=0.0), _row(1, 41.0, err=None)]
    text = results_module._estimate_text
    assert all(text(row, 'radius').endswith('-') for row in _assess(rows, 'index'))


def synthetic_series(rows):
    return SeriesResult(
        rows,
        variable='index',
        template={'configuration': {'model_name': 'sphere'}},
        start='previous',
        engine='bumps',
        method=None,
        fit_options={},
        planned=len(rows),
    )


def test_long_tables_are_abbreviated():
    text = str(synthetic_series([_row(i, 40.0 + i) for i in range(30)]))
    labels = [line.split()[1] for line in text.splitlines() if line[:4].strip().isdigit()]
    assert labels == [f'r{i}' for i in (*range(10), *range(20, 30))]
    assert 'Showing 20 of 30 rows' in text


def test_flagged_is_a_read_only_selection_keyed_by_original_row(tmp_path):
    rows = [_row(i, 40.0 + i) for i in range(10)]
    rows[6] = _row(6, 90.0)
    flagged = synthetic_series(rows).flagged()
    assert [row.index for row in flagged] == [6, 7]
    assert flagged[6].flag_names == ('jump',)
    with pytest.raises(KeyError):
        flagged[0]
    with pytest.raises(ValueError, match='read-only'):
        flagged.update(6, IQuana())
    with pytest.raises(ValueError, match='read-only'):
        flagged.save_analysis(str(tmp_path / 'x.json'))
    assert '2 of 10 rows flagged' in str(flagged)


def test_update_replaces_a_row_and_recomputes_flags(tmp_path):
    for i in range(3):
        write_dataset(simulate(60.0, seed=i), tmp_path / f'r{i}.dat')
    template = make_template()
    template.set_param('radius', value=40, min=10, max=50)
    series = template.fit_series(str(tmp_path / 'r*.dat'))
    assert len(series.flagged()) == 3

    for row in list(series):
        repaired = series.fitter(row.index)
        repaired.set_param('radius', max=200)
        repaired.fit()
        series.update(row.index, repaired)
    assert len(series.flagged()) == 0
    assert all(row.edited for row in series)
    assert series[0].estimate('radius')[0] == pytest.approx(60.0, abs=2)
    assert 'edited' in str(series)


def test_update_refuses_a_stale_result_other_data_or_another_model(fitted, scan):
    before = fitted[2]
    stale = fitted.fitter(2)
    stale.set_param('radius', max=99)
    with pytest.raises(ValueError, match='Fit this dataset again before updating row 2'):
        fitted.update(2, stale)

    other = fitted.fitter(3)
    with pytest.raises(ValueError, match='does not hold the data of row 2'):
        fitted.update(2, other)

    cylinder = IQuana()
    cylinder.load_data(fitted[2].source)
    cylinder.set_model('cylinder')
    cylinder.set_resolution('none')
    cylinder.set_param('radius', value=40, min=10, max=100, vary=True)
    cylinder.fit()
    with pytest.raises(ValueError, match='different model'):
        fitted.update(2, cylinder)
    assert fitted[2] == before


def test_save_and_load_round_trip_without_the_data(fitted, tmp_path, scan):
    path = str(tmp_path / 'scan.json')
    fitted.save_analysis(path)
    shutil.rmtree(scan)
    reloaded = SeriesResult.load_analysis(path)
    assert reloaded.table() == fitted.table()
    assert str(reloaded) == str(fitted)


def test_a_moved_series_finds_its_data_and_saves_again_elsewhere(tmp_path):
    project = tmp_path / 'project'
    write_scan(str(project / 'data'))
    series = make_template().fit_series(str(project / 'data' / 'T_{temperature}K.dat'))
    series.save_analysis(str(project / 'scan.json'))

    moved = tmp_path / 'moved'
    shutil.move(str(project), str(moved))
    reloaded = SeriesResult.load_analysis(str(moved / 'scan.json'))
    assert reloaded.fitter(1).data is not None

    # Saved again at another depth, the relative paths are re-based on the new file.
    (moved / 'copies').mkdir()
    reloaded.save_analysis(str(moved / 'copies' / 'scan.json'))
    shutil.move(str(moved), str(tmp_path / 'third'))
    again = SeriesResult.load_analysis(str(tmp_path / 'third' / 'copies' / 'scan.json'))
    assert again.fitter(0).data is not None


def test_a_changed_data_file_keeps_the_numbers_and_draws_a_theory_curve(fitted, tmp_path):
    path = str(tmp_path / 'scan.json')
    fitted.save_analysis(path)
    write_dataset(simulate(70.0, seed=99), fitted[1].source)
    reloaded = SeriesResult.load_analysis(path)
    assert reloaded[1].estimate('radius') == fitted[1].estimate('radius')
    with pytest.warns(UserWarning, match='data has changed'):
        figure = reloaded.plot_fits(rows=[1], show=False)
    assert any('theory at saved values' in (trace.name or '') for trace in figure.data)


def test_plot_fits_names_rows_it_cannot_draw(scan):
    with open(os.path.join(scan, 'T_025K.dat'), 'w') as handle:
        handle.write('not data\n')
    series = make_template().fit_series(os.path.join(scan, 'T_{temperature}K.dat'))
    bad = series['T_025K.dat'].index
    with pytest.warns(UserWarning, match=f'#{bad} T_025K.dat: failed'):
        series.plot_fits(rows=[0, bad], show=False)
    with pytest.raises(ValueError, match='None of the rows can be drawn'):
        series.plot_fits(rows=[bad], show=False)
    assert len(series.plot_fits(show=False).data) == 2 * len(TEMPERATURES)


def test_the_trend_plot_has_a_panel_per_parameter_and_marks_failures(scan):
    with open(os.path.join(scan, 'T_025K.dat'), 'w') as handle:
        handle.write('not data\n')
    series = make_template().fit_series(os.path.join(scan, 'T_{temperature}K.dat'))
    figure = series.plot(show=False)
    assert 'failed' in [trace.name for trace in figure.data]
    assert '1 failed' in figure.layout.title.text
    radius_only = series.plot('radius', show=False)
    assert len({trace.yaxis for trace in radius_only.data}) == 2  # radius, then χ²
    with pytest.raises(ValueError, match='Unknown parameter'):
        series.plot('nope')


def test_custom_models_and_newer_rows_are_refused(fitted, tmp_path):
    path = tmp_path / 'scan.json'
    fitted.save_analysis(str(path))
    document = json.loads(path.read_text())
    document['rows'][0]['analysis']['configuration']['model_name'] = 'custom.mine'
    path.write_text(json.dumps(document))
    with pytest.raises(AnalysisFileError, match='allow_custom_models'):
        SeriesResult.load_analysis(str(path)).fitter(0)

    document['rows'][0]['analysis']['schema']['version'] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(AnalysisFileError, match=r'rows\[0\]\.analysis\.schema\.version 99'):
        SeriesResult.load_analysis(str(path))


def test_the_wrong_loader_names_the_right_one(fitted, tmp_path, scan):
    path = str(tmp_path / 'scan.json')
    fitted.save_analysis(path)
    with pytest.raises(AnalysisFileError, match=r'Use SeriesResult\.load_analysis'):
        IQuana.load_analysis(path)
    single = make_template(os.path.join(scan, 'T_020K.dat'))
    single.save_analysis(str(tmp_path / 'one.json'))
    with pytest.raises(AnalysisFileError, match=r'Use IQuana\.load_analysis'):
        SeriesResult.load_analysis(str(tmp_path / 'one.json'))


def test_an_interrupted_run_saves_and_reloads_as_such(pattern, monkeypatch, tmp_path, capsys):
    real_fit = IQuana.fit
    calls = []

    def interrupted_fit(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(IQuana, 'fit', interrupted_fit)
    with pytest.warns(UserWarning):
        series = make_template().fit_series(pattern)
    path = str(tmp_path / 'part.json')
    series.save_analysis(path)
    capsys.readouterr()
    reloaded = SeriesResult.load_analysis(path)
    assert '3 of 6 processed, interrupted' in capsys.readouterr().out
    assert '3 of 6 processed, interrupted' in str(reloaded)


def test_warnings_are_collected_per_row_not_printed(tmp_path, scan):
    template = make_template()
    template.set_resolution('data')  # the files carry no dQ: one warning per row
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        series = template.fit_series(os.path.join(scan, 'T_{temperature}K.dat'))
    texts = series.warnings()
    assert len(texts) == 1 and next(iter(texts.values())) == len(TEMPERATURES)


def test_a_row_explains_itself(fitted):
    text = repr(fitted[3])
    assert text.startswith('Row 3')
    assert 'started from: row 2 in the original run' in text
    assert 'radius' in text and 'series.fitter(3)' in text
    assert 'started from: the template' in repr(fitted[0])
