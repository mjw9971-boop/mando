"""논문 판형 표·그림 (tools/paper_figs.py) 과 2026-10-03 집계 기준.

지키는 것:
  · 집계 기준 — 지나감 횡거리 8 m, 종료선(report.json finish_thr_m) 이후 시작한 충돌은 세지
    않는다. 뺀 것은 collisions_after_finish.csv 에 남는다. --after-finish include 면 이전 집계.
  · 실패 원인 분류는 로그에서 읽는 규칙이다 (시나리오 이름을 적지 않는다):
    기반 실패 → 시뮬레이터, 오탐 시프트 → 오탐, 움직이는 차량이 거의 선 자차로 들어옴 ·
    종료선 콘 → 시나리오, 나머지 → 제어기 판단.
  · 같은 명령에 --arm 을 하나 더 주면 비교판이 나온다 (표 열 · 막대 · 선이 구현마다).
  · 그림은 한 단 폭 8.5 cm, PNG 600 dpi + PDF + 원천 CSV.
"""
import json
import math
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))

import paper_figs as pf                                         # noqa: E402
import paper_report as pr                                       # noqa: E402
from test_paper_report import (CAR, OBS_X, _car, _car_xml, _log_hit, _log_pass, _rows,  # noqa: E402
                               _tick, _write_arm)


def _scen(scen, typ, name, events, xml, axes=None):
    d = scen / typ
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.yaml').write_text(yaml.safe_dump({'name': name, 'theme': typ, 'events': events,
                                                    'axes': axes or {}}, allow_unicode=True), encoding='utf-8')
    (d / f'{name}.xml').write_text(f'<Scenario>{xml}</Scenario>', encoding='utf-8')


def _log_pass_then_cut_in():
    """정차차를 비켜 지나 x=160 부터 1 m/s 로 기어가는데, 옆에서 움직이는 차가 꺾여 들어와 닿는다.

    자차는 계속 전진한다 — 채점기의 driving_span 은 route_s 가 최고점에 처음 닿은 틱에서 끊으므로
    (완주 런), 선 채로 닿으면 종료선 규칙과 무관하게 이미 안 센다.
    """
    out = _log_pass()
    i0 = len(out)
    for k in range(40):
        y = 3.0 - 0.1 * k                               # 옆 차로에서 자차 쪽으로
        x = 160.0 + 0.05 * k
        out.append(_tick(i0 + k, x, 0.0, 1.0, [(7, x + 2.0, y, 3.0, CAR[0], CAR[1], 'vehicle')]))
        out[-1]['raw']['objects'][0][4] = -0.6          # 진행각 −34°
    return out


# ── 종료선 이후 제외 ─────────────────────────────────────────────────────
@pytest.mark.parametrize('mode,success,n_after', [('exclude', 'True', 1), ('include', 'False', 0)])
def test_collision_after_finish(tmp_path, mode, success, n_after):
    scen = tmp_path / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    _scen(scen, '회피', '회피_01', ev, _car_xml('StoppedCar01', OBS_X, 0.0))
    _write_arm(tmp_path, 'A', '회피', [('회피_01', '완주', _log_pass_then_cut_in(), {'collision': 1})],
               finish_thr_m=150.0)   # report.json(채점기)은 종료선 이후도 센다
    out = tmp_path / 'out'
    assert pr.main(['--arm', f'A={tmp_path / "A"}', '--scenarios', str(scen), '--out', str(out),
                    '--no-figures', '--arms-yaml', str(tmp_path / 'none.yaml'), '--after-finish', mode]) == 0
    run = _rows(out / 'runs.csv')[0]
    assert run['success'] == success
    assert len(_rows(out / 'collisions_after_finish.csv')) == n_after
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['thresholds']['after_finish'] == mode and meta['thresholds']['pass_lat_m'] == 8.0
    assert not any('충돌 수 불일치' in w for w in meta['warnings'])


def test_contact_onset_reports_who_moved():
    veh = {'length': 4.6, 'width': 1.9, 'wheelbase': 2.7, 'front_overhang_m': 1.0}
    span = _log_pass_then_cut_in()[-40:]
    c = pr.contact_onset(span, 7, veh, 0.1, span[0]['t'])
    assert c['ego_v_mps'] == 1.0 and c['obj_v_mps'] == 3.0 and c['obj_dyaw_deg'] == pytest.approx(-34.4, abs=0.1)


# ── 실패 원인 분류 ───────────────────────────────────────────────────────
def _run(status='완주', success=False, why='collision', cols=(), shifts=(), queue=None):
    return {'status': status, 'success': success, 'why': why,
            'an': {'collisions': list(cols), 'shifts': list(shifts), 'queue': queue, 'snapshot': {}}}


def _col(kind='event', ego=6.0, obj=0.0, dyaw=0.0):
    return {'obj_kind': kind, 'ego_v_mps': ego, 'obj_v_mps': obj, 'obj_dyaw_deg': dyaw, 'min_gap_m': 0.0}


def test_failure_cause_rules():
    assert pr.failure_cause(_run(status='VTD 9910 무응답'))[0] == 'simulator'
    assert pr.failure_cause(_run(success=True))[0] is None
    fp = {'blocker_kind': 'queue', 't': 3.0, 's_rel_m': 79.0, 'light_state': 'red', 'false_positive': True}
    assert pr.failure_cause(_run(why='false_positive_shift', queue={'shifts_on_queue': [fp]}))[0] == 'false_positive'
    # 움직이는 차량이 꺾여 거의 선 자차로 → 시나리오
    assert pr.failure_cause(_run(cols=[_col('queue', ego=0.8, obj=3.0, dyaw=22)]))[0] == 'scenario'
    assert pr.failure_cause(_run(cols=[_col('finish_cone', ego=5.0)]))[0] == 'scenario'
    # 자차가 달리다 정차차에 닿음 · 자차가 옆 차로 교통(나란히 가는 차)으로 들어감 → 제어기
    assert pr.failure_cause(_run(cols=[_col('event', ego=6.0)]))[0] == 'controller'
    assert pr.failure_cause(_run(cols=[_col('other', ego=2.5, obj=2.8, dyaw=-6)]))[0] == 'controller'
    # 둘 중 하나라도 제어기 쪽이면 제어기
    assert pr.failure_cause(_run(cols=[_col('queue', ego=0.8, obj=3.0, dyaw=22), _col('event')]))[0] == 'controller'
    assert pr.failure_cause(_run(why='not_passed(완주)'))[0] == 'controller'
    assert pr.failure_cause(_run(status='blocked', why='inf_stop(blocked)'))[0] == 'controller'


# ── paper_figs: 한 구현 → 같은 명령에 구현 하나 더 ──────────────────────
@pytest.fixture(scope='module')
def figs_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('figs')
    scen = root / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    names = ['근접_01', '근접_02', '근접_03']
    for n, d in zip(names, (8, 15, 30)):
        _scen(scen, '근접', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0), axes={'거리': d})
    (scen / 'batch_근접.json').write_text(json.dumps([{'name': n} for n in names]), encoding='utf-8')
    for arm, runs in (('A', [('근접_01', '완주', _log_pass(), {}), ('근접_02', '완주', _log_pass(), {}),
                             ('근접_03', 'SCP 실패', None, None)]),
                      ('B', [('근접_01', '완주', _log_pass(), {}), ('근접_02', 'stall', _log_hit(), {'collision': 1}),
                             ('근접_03', '완주', _log_pass(), {})])):
        _write_arm(root, arm, '근접', runs)
    arms_yaml = root / 'arms.yaml'
    arms_yaml.write_text(yaml.safe_dump({'report': {
        'reference': 'A', 'labels': {'A': {'label_ko': '최종 구현'}, 'B': {'label_ko': '초기 구현'}},
        'paper_figs': {'trace_type': '근접', 'trace_axis': '거리', 'compare_arm': 'B'}}},
        allow_unicode=True), encoding='utf-8')
    return root, scen, arms_yaml


def _figs(root, scen, arms_yaml, arms, out):
    argv = sum((['--arm', f'{a}={root / a}'] for a in arms), []) + [
        '--scenarios', str(scen), '--out', str(out), '--arms-yaml', str(arms_yaml)]
    return pf.main(argv)


def test_one_arm_then_comparison(figs_inputs, tmp_path):
    root, scen, arms_yaml = figs_inputs
    if pf.setup_font() is None:
        pytest.skip('한글 폰트 없음')
    one, two = tmp_path / 'one', tmp_path / 'two'
    assert _figs(root, scen, arms_yaml, ['A'], one) == 0
    assert _figs(root, scen, arms_yaml, ['A', 'B'], two) == 0
    # 표 2 — 구현마다 열 묶음. 기반 실패는 분모에서 빠진다.
    t1 = {(r['type'], r['arm']): r for r in _rows(one / 'table2_paper.csv')}
    assert t1[('근접', 'A')]['n_judged'] == '2' and t1[('근접', 'A')]['success_pct'] == '100.0'
    md2 = (two / 'table2_paper.md').read_text(encoding='utf-8')
    assert '최종 구현 주 지표' in md2 and '초기 구현 주 지표' in md2
    t2 = {(r['type'], r['arm']): r for r in _rows(two / 'table2_paper.csv')}
    assert t2[('근접', 'B')]['success'] == '2' and t2[('근접', 'B')]['collision_runs'] == '1'
    assert t2[('근접', 'B')]['inf_stop'] == '1'
    # 그림: 한 단 폭 · 600 dpi · PDF · 원천 CSV
    from PIL import Image
    for d in (one, two):
        for stem in ('fig_A_success_by_type', 'fig_B_near_stop_trace', 'fig_C_failure_causes'):
            im = Image.open(d / f'{stem}.png')
            assert round(im.size[0] / 600 * 2.54, 1) == 8.5 and round(im.info['dpi'][0]) == 600
            assert (d / f'{stem}.pdf').stat().st_size > 0 and (d / f'{stem}.csv').exists()
    # 그림 B: 모든 구현이 끝까지 간 런 중 기준 구현이 성공한 것, 시작 거리가 가장 짧은 것
    b2 = _rows(two / 'fig_B_near_stop_trace.csv')
    assert {r['arm'] for r in b2} == {'A', 'B'} and {r['scenario'] for r in b2} == {'근접_01'}
    meta_b = json.loads((two / 'paper_figs_meta.json').read_text(encoding='utf-8'))['trace']
    assert set(meta_b['start_gap_m']) == {'A', 'B'} and meta_b['axis_value'] == 8
    first = {r['arm']: float(r['gap_m']) for r in reversed(b2)}       # 각 구현의 t=0 행
    assert all(meta_b['start_gap_m'][a] == pytest.approx(first[a]) for a in ('A', 'B'))
    # 그림 C: 원인별 런 수 — A 는 기반 실패 1, B 는 정차차 충돌 1 (제어기 판단)
    c2 = {(r['arm'], r['cause']): int(r['runs']) for r in _rows(two / 'fig_C_failure_causes.csv')}
    assert c2[('A', 'simulator')] == 1 and c2[('B', 'controller')] == 1 and sum(c2.values()) == 2
    # Ⅳ.2 초안 — 비교 구현이 없으면 '—', 있으면 수치
    d1 = (one / 'IV2_draft.md').read_text(encoding='utf-8')
    d2 = (two / 'IV2_draft.md').read_text(encoding='utf-8')
    assert '초기 구현의 성공률은 —' in d1 and '초기 구현의 성공률은 2/3' in d2
    assert '## 출처' in d2


def test_type_label():
    assert pf.type_label('논문2_근접정지', {}) == '근접정지'
    assert pf.type_label('논문7_오탐_급정거', {}) == '오탐 급정거'
    assert pf.type_label('논문2_근접정지', {'논문2_근접정지': '근접 정지'}) == '근접 정지'
    assert pf.type_label('기타', {}) == '기타'


def test_cff_font_detection(tmp_path):
    otf, ttf = tmp_path / 'a.otf', tmp_path / 'b.ttf'
    otf.write_bytes(b'OTTO' + b'\0' * 12)
    ttf.write_bytes(b'\0\1\0\0' + b'\0' * 12)
    assert pf._is_cff(str(otf)) and not pf._is_cff(str(ttf))
    assert not math.isnan(pf.COL_W) and round(pf.COL_W * 2.54, 2) == 8.5


def test_primary_metric_is_false_positive_runs_for_fp_types(tmp_path):
    """2026-10-03 결정 3: 오탐 유형은 오탐 런이 주 지표 (성공률은 참고)."""
    rep = tmp_path / 'report'
    rep.mkdir()
    head = ('type,class,arm,n_expected,n_valid,n_invalid,n_judged,n_excluded,success,success_pct,'
            'success_lo_pct,success_hi_pct,inf_stop,timeout,runs_with_collision,collisions,false_positive_runs\n')
    (rep / 'by_type.csv').write_text(head + '회피,avoid,A,10,10,0,10,0,9,90.0,59.6,98.2,0,0,1,1,0\n'
                                     '줄,queue,A,10,10,0,10,0,2,20.0,5.7,51.0,0,0,5,6,4\n', encoding='utf-8')
    rows = pf.table2(rep, ['A'], lambda a: a, lambda t: t)
    q = next(r for r in rows if r['type'] == '줄')
    v = next(r for r in rows if r['type'] == '회피')
    assert (q['primary_metric'], q['primary_k'], q['primary_n'], q['primary_pct']) == ('false_positive_runs', 4, 10, 40.0)
    assert (q['primary_lo_pct'], q['primary_hi_pct']) == (16.8, 68.7)
    assert (v['primary_metric'], v['primary_pct']) == ('success', 90.0)
    md = pf.table2_md(rows, ['A'], lambda a: a)
    assert '오탐 4/10 런 = 40.0% (16.8–68.7)' in md and '성공률 20.0%' in md and '성공 90.0% (59.6–98.2)' in md
