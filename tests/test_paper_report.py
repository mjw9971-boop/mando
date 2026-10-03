"""tools/paper_report.py — 논문 Ⅳ장 집계기.

지키는 것 (2026-10-02):
  · 수치는 report.json·로그에서만 나온다 — 여기서는 손으로 만든 작은 로그로
    "무엇을 세는가" 를 고정한다 (무한 정지 = stall + blocked, timeout 은 따로,
    기반 실패는 분모에서 뺀다, 충돌은 채점기와 같은 검출기).
  · 입력은 "구현 이름 = 로그 폴더" 목록이다. arms.yaml 에 없는 구현(다른 브랜치의
    초기 구현)도 폴더만 맞으면 합쳐진다.
  · 급정거 오탐 = 시프트를 만든 대상 객체가 그 뒤 정지 임계를 넘겨 움직인 경우.
  · 일부 구현·일부 런만 있는 중간 상태에서도 돈다.
"""
import csv
import json
import math
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import paper_report as pr                                       # noqa: E402

DT = 0.05
CAR = (4.4, 1.8)            # 정차차 길이·폭
OBS_X = 100.0               # 정차차 x


# ── 통계 ─────────────────────────────────────────────────────────────────
def test_wilson():
    lo, hi = pr.wilson(0, 1)
    assert lo == 0.0 and hi == pytest.approx(0.7935, abs=1e-3)
    lo, hi = pr.wilson(15, 15)
    assert lo == pytest.approx(0.7961, abs=1e-3) and hi == 1.0
    lo, hi = pr.wilson(30, 60)
    assert lo == pytest.approx(0.3773, abs=1e-3) and hi == pytest.approx(0.6227, abs=1e-3)
    assert all(math.isnan(v) for v in pr.wilson(0, 0))


def test_mcnemar_exact():
    assert pr.mcnemar_exact(0, 0) == 1.0
    assert pr.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert pr.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert pr.mcnemar_exact(3, 3) == 1.0
    assert pr.mcnemar_exact(10, 1) == pytest.approx(2 * (1 + 11) / 2 ** 11)


# ── 손으로 만든 시나리오·로그 ────────────────────────────────────────────
def _scenario(scen, typ, name, events, xml_body):
    d = scen / typ
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.yaml').write_text(yaml.safe_dump(
        {'name': name, 'theme': typ, 'events': events, 'axes': {}}, allow_unicode=True), encoding='utf-8')
    (d / f'{name}.xml').write_text(f'<Scenario>{xml_body}</Scenario>', encoding='utf-8')


def _car_xml(name, x, y):
    return (f'<Player><Description Name="{name}"/><Init><PosAbsolute X="{x}" Y="{y}"/></Init></Player>')


def _tick(i, x, y, v, objs, avoid=None, light=None, v_target=None, dsl=None):
    """objs: [(id, x, y, speed, 길이, 폭, cls)]"""
    return {'t': 1000.0 + i * DT,
            'raw': {'ego': [x, y, 0, 0, 0, 0], 'lights': [],
                    'objects': [[oid, ox, oy, 0.0, 0.0, sp, ln, wd, 1.5] for oid, ox, oy, sp, ln, wd, _c in objs]},
            'ego': {'x': x, 'y': y, 'yaw': 0.0, 'speed': v, 'route_s': x, 'lane': [1, 0, -1],
                    's': x, 't_off': 0.0},
            'world': {'light': light, 'summ': {'dist_stop_line': dsl, 'in_junction': False}},
            'objects': [{'id': oid, 'x': ox, 'y': oy, 'speed': sp, 'cls': c}
                        for oid, ox, oy, sp, _l, _w, c in objs],
            'decision': {'state': 'none', 'v_target': v if v_target is None else v_target,
                         'reasons': {'winner': 'none', 'avoid': avoid, 'escape': None}}}


def _car(x=OBS_X, y=0.0, sp=0.0, oid=2):
    return (oid, x, y, sp, CAR[0], CAR[1], 'vehicle')


def _log_pass():
    """정차차를 옆(3.2 m)으로 비켜 지나 x=160 까지."""
    out = []
    for i in range(320):
        x = 0.5 * i
        y = 3.2 if 80.0 <= x <= 125.0 else 0.0
        out.append(_tick(i, x, y, 10.0, [_car()] if abs(OBS_X - x) < 80 else []))
    return out


def _log_stop(light=None, extra=()):
    """정차차 12 m 앞(뒷축 기준)에서 서서 60 s."""
    out = [_tick(i, 0.5 * i, 0.0, 10.0, [_car(), *extra] if 0.5 * i > 20 else []) for i in range(177)]
    out += [_tick(177 + k, 88.0, 0.0, 0.0, [_car(), *extra], light=light, v_target=0.0, dsl=40.0)
            for k in range(int(60 / DT))]
    return out


def _log_hit():
    """정차차에 그대로 닿아 선다 (범퍼 간격 0)."""
    stop_x = OBS_X - CAR[0] / 2 - 3.8          # 앞범퍼 = 뒷축 + 3.8 (wheelbase + 앞 오버행 ≈ 3.8)
    out = [_tick(i, min(0.5 * i, stop_x), 0.0, 5.0, [_car()]) for i in range(200)]
    out += [_tick(200 + k, stop_x, 0.0, 0.0, [_car()], v_target=3.0) for k in range(int(60 / DT))]
    return out


def _write_arm(root, arm, typ, runs, finish_thr_m=1e6):
    """runs: [(이름, status, ticks 또는 None, violations 또는 None)]

    finish_thr_m: report.json 의 종료선 route_s (batch_run 이 쓴다). 기본은 멀리 — 종료선
    이후 제외가 끼어들지 않게.
    """
    d = root / arm / typ
    d.mkdir(parents=True, exist_ok=True)
    rep = []
    for name, status, ticks, vio in runs:
        res = {'name': name, 'status': status, 'log': None, 'finish_thr_m': finish_thr_m}
        if ticks is not None:
            p = d / f'{name}.jsonl'
            p.write_text(''.join(json.dumps(t) + '\n' for t in ticks), encoding='utf-8')
            res['log'] = '/다른PC/경로/' + p.name          # 절대경로가 안 맞아도 폴더에서 찾는다
        if vio is not None:
            res['violations'] = vio
        rep.append(res)
    (d / 'report.json').write_text(json.dumps(rep, ensure_ascii=False), encoding='utf-8')


def _rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp('paper')
    scen = root / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    for n in ('회피_01', '회피_02', '회피_03'):
        _scenario(scen, '회피', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0))
    (scen / 'batch_회피.json').write_text(json.dumps([{'name': n} for n in ('회피_01', '회피_02', '회피_03')]))
    _scenario(scen, '급정거', '급정거_01', [{'kind': 'lead_brake', 'lead': 'LeadCar01', 'resume': True, 'decel': 5.0}], '')
    _scenario(scen, '급정거', '급정거_02', [{'kind': 'lead_brake', 'lead': 'LeadCar01', 'resume': False, 'decel': 5.0}], '')
    (scen / 'batch_급정거.json').write_text(json.dumps([{'name': '급정거_01'}, {'name': '급정거_02'}]))

    red = [7, 1]
    queue_car = (9, 96.0, 0.0, 0.0, 4.4, 1.8, 'vehicle')          # 시나리오 장애물이 아닌 앞차
    # REF: 전부 완주. X1: 정지 2 (하나는 적신호 앞 큐 의심) + 기반 실패 1. X4: 충돌+stall 1, timeout 1.
    _write_arm(root, 'REF', '회피', [('회피_01', '완주', _log_pass(), {}),
                                    ('회피_02', '완주', _log_pass(), {}),
                                    ('회피_03', '완주', _log_pass(), {})])
    _write_arm(root, 'X1', '회피', [('회피_01', 'blocked', _log_stop(), {}),
                                   ('회피_02', 'blocked', _log_stop(light=red, extra=(queue_car,)), {}),
                                   ('회피_03', 'SCP 실패: boom', None, None)])
    _write_arm(root, 'X4', '회피', [('회피_01', 'stall', _log_hit(), {'collision': 1}),
                                   ('회피_02', 'timeout', _log_stop(), {})])
    # 급정거: 앞차(id 5)가 섰다가 다시 출발 — 그 사이에 시프트를 만들면 오탐
    def brake(moves_again):
        out = []
        for i in range(300):
            sp = 0.0 if 100 <= i < 200 or not moves_again and i >= 100 else 8.0
            lead = (5, 60.0 + (0.4 * min(i, 100)), 0.0, sp, 4.4, 1.8, 'vehicle')
            av = None
            if i == 150:
                av = {'state': 'PREEMPT', 'shift': 'left', 'blocker': 5, 'span': [100, 400],
                      'span_new': [100, 400], 'chain': [5], 's_rel': 30.0, 'trigger': 'corridor'}
            elif 150 < i < 220:
                av = {'state': 'SHIFT_ACTIVE', 'span': [100, 400]}
            out.append(_tick(i, 0.3 * i, 0.0, 6.0, [lead], avoid=av))
        return out
    _write_arm(root, 'REF', '급정거', [('급정거_01', '완주', brake(True), {}),
                                      ('급정거_02', '완주', brake(False), {})])
    out = root / 'out'
    rc = pr.main(['--arm', f'REF={root / "REF"}', '--arm', f'X1={root / "X1"}', '--arm', f'X4={root / "X4"}',
                  '--arm', f'NONE={root / "없는폴더"}', '--scenarios', str(scen), '--out', str(out),
                  '--ref', 'REF', '--arms-yaml', str(root / 'no_arms.yaml'),
                  '--label-ko', 'X1=시험 구현'])
    assert rc == 0
    return out


def test_table2_counts(report):
    t = {r['arm']: r for r in _rows(report / 'table2.csv')}
    assert set(t) == {'REF', 'X1', 'X4', 'NONE'}
    ref, x1, x4 = t['REF'], t['X1'], t['X4']
    assert (ref['n_valid'], ref['completed'], ref['completion_pct'], ref['inf_stop']) == ('3', '3', '100.0', '0')
    # 기반 실패는 분모에서 빠지고 따로 센다
    assert (x1['n_valid'], x1['n_invalid'], x1['n_expected']) == ('2', '1', '3')
    assert (x1['inf_stop'], x1['blocked'], x1['stall'], x1['timeout'], x1['completion_pct']) == ('2', '2', '0', '0', '0.0')
    assert x1['label'] == '시험 구현'
    # timeout 은 무한 정지에 들어가지 않는다
    assert (x4['inf_stop'], x4['stall'], x4['timeout'], x4['n_valid']) == ('1', '1', '1', '2')
    assert (x4['collisions'], x4['runs_with_collision'], x4['collisions_event_obstacle']) == ('1', '1', '1')
    assert float(x4['min_gap_m']) <= 0.1
    assert t['NONE']['n_valid'] == '0' and t['NONE']['completion_pct'] == ''
    md = (report / 'table2.md').read_text(encoding='utf-8')
    assert '무한 정지 [회]' in md and 'timeout [회]' in md and '95% Wilson' in md


def test_per_encounter(report):
    wide = {r['scenario']: r for r in _rows(report / 'per_encounter.csv')}
    assert len(wide) == 3
    assert wide['회피_01']['REF_outcome'] == 'success' and wide['회피_01']['X1_outcome'] == 'stopped'
    assert wide['회피_01']['X4_outcome'] == 'collision'
    assert wide['회피_03']['X1_outcome'] == '' and wide['회피_03']['REF_outcome'] == 'success'   # 안 돈 것은 빈칸
    assert float(wide['회피_01']['REF_min_gap_m']) > 1.0
    col = _rows(report / 'collisions.csv')
    assert [(c['arm'], c['scenario'], c['obj_kind'], c['encounter']) for c in col] == [('X4', '회피_01', 'event', '0')]


def test_stall_review_marks_queue_suspect(report):
    sr = {(r['arm'], r['scenario']): r for r in _rows(report / 'stall_review.csv')}
    assert set(sr) == {('X1', '회피_01'), ('X1', '회피_02'), ('X4', '회피_01')}      # timeout 은 없다
    plain, queue = sr[('X1', '회피_01')], sr[('X1', '회피_02')]
    assert plain['light_state'] == '' and plain['queue_suspect'] == 'False'
    assert plain['ahead_kind'] == 'event' and float(plain['ahead_lon_m']) == pytest.approx(OBS_X - 88.0 - 3.799, abs=0.2)
    assert float(plain['stopped_s']) == pytest.approx(60.0, abs=0.2)
    assert queue['light_state'] == 'red' and queue['light_hold_frac'] == '1.0'
    assert queue['ahead_kind'] == 'other' and queue['ahead_id'] == '9' and queue['queue_suspect'] == 'True'
    assert queue['dist_stop_line_m'] == '40.0' and queue['nearest_encounter_dist_m'] == '12.0'


def test_paired_mcnemar(report):
    p = {(r['unit'].split('(')[0], r['other']): r for r in _rows(report / 'paired.csv')}
    s1 = p[('scenario', 'X1')]
    assert (s1['ref'], s1['n_pairs'], s1['ref_only_success'], s1['other_only_success']) == ('REF', '2', '2', '0')
    assert float(s1['mcnemar_exact_p']) == pytest.approx(0.5)
    assert p[('scenario', 'NONE')]['n_pairs'] == '0'


def test_by_type_covers_every_class_with_its_own_criterion(report):
    bt = {(r['type'], r['arm']): r for r in _rows(report / 'by_type.csv')}
    assert {t for t, _a in bt} == {'회피', '급정거'}
    assert bt[('회피', 'REF')]['class'] == 'avoid' and bt[('급정거', 'REF')]['class'] == 'brake'
    # 회피: 지나감 + 충돌 없음 + 무한 정지 없음
    assert (bt[('회피', 'REF')]['success'], bt[('회피', 'REF')]['n_judged']) == ('3', '3')
    assert (bt[('회피', 'X1')]['success'], bt[('회피', 'X1')]['n_judged']) == ('0', '2')
    assert (bt[('회피', 'X4')]['success'], bt[('회피', 'X4')]['n_judged']) == ('0', '2')
    # 급정거: 다시 움직인 앞차에 시프트 = 오탐 (2런 중 1런)
    b = bt[('급정거', 'REF')]
    assert (b['success'], b['n_judged'], b['false_positive_runs']) == ('1', '2', '1')
    meta = json.loads((report / 'meta.json').read_text(encoding='utf-8'))
    assert meta['avoid_types'] == ['회피'] and meta['brake_types'] == ['급정거']
    assert meta['reference'] == 'REF'
    t2 = {r['arm']: r for r in _rows(report / 'table2.csv')}       # 표 2 는 회피 유형만
    assert (t2['REF']['success_pct'], t2['X1']['success_pct']) == ('100.0', '0.0')


def test_failures_record_why_and_distance_to_obstacle(report):
    fl = {(r['arm'], r['scenario']): r for r in _rows(report / 'failures.csv')}
    assert set(fl) == {('X1', '회피_01'), ('X1', '회피_02'), ('X4', '회피_01'), ('X4', '회피_02'),
                       ('REF', '급정거_01')}
    stop = fl[('X1', '회피_01')]
    assert stop['why'] == 'inf_stop(blocked)' and stop['failed_encounter_outcome'] == 'stopped'
    assert float(stop['obstacle_gap_m']) == pytest.approx(OBS_X - 88.0 - 3.799 - CAR[0] / 2, abs=0.2)
    assert fl[('X4', '회피_01')]['why'] == 'inf_stop(stall)' and fl[('X4', '회피_01')]['collision_obj_kind'] == 'event'
    assert fl[('X4', '회피_02')]['why'] == 'not_passed(timeout)'      # 무한 정지는 아니지만 못 지나갔다
    assert fl[('REF', '급정거_01')]['why'] == 'false_positive_shift'


def test_brake_false_positive(report):
    rows = {r['scenario']: r for r in _rows(report / 'brake_fp.csv')}
    assert set(rows) == {'급정거_01', '급정거_02'}                 # 시프트는 런마다 한 번만 센다
    a, b = rows['급정거_01'], rows['급정거_02']
    assert (a['blocker_id'], a['tick'], a['false_positive'], a['lead_resume']) == ('5', '150', 'True', 'True')
    assert float(a['blocker_speed_at_shift']) == 0.0 and float(a['blocker_speed_max_after']) == 8.0
    assert float(a['t_moved_s']) == pytest.approx(200 * DT)
    assert (b['false_positive'], b['t_moved_s'], b['lead_resume']) == ('False', '', 'False')
    s = {r['arm']: r for r in _rows(report / 'brake_fp_summary.csv')}
    assert (s['REF']['runs'], s['REF']['shifts'], s['REF']['false_positive_shifts'],
            s['REF']['runs_with_false_positive']) == ('2', '2', '1', '1')
    assert s['X1']['runs'] == '0'


def test_figures_and_sources(report):
    f = report / 'figures'
    for stem in ('fig_table2', 'fig_by_type', 'fig_end_reason', 'fig_brake_fp'):
        assert (f / f'{stem}.csv').exists() and (f / f'{stem}.png').stat().st_size > 1000, stem
    er = _rows(f / 'fig_end_reason.csv')
    assert {(r['arm'], r['end_reason']): r['count'] for r in er}[('X1', 'invalid')] == '1'


def test_nested_batches_are_merged_and_done_run_wins(tmp_path):
    """--out-dir/--resume 이 없는 브랜치: logs/batch/<시각>/ 을 유형 폴더 아래 통째로 옮겨 넣는다."""
    scen = tmp_path / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    for n in ('회피_01', '회피_02'):
        _scenario(scen, '회피', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0))
    arm = tmp_path / 'WON'
    # 1차 배치: 01 완주, 02 는 기반 실패.  2차 배치: 02 만 다시 — blocked.
    _write_arm(arm / '회피', '.', '20261002_100000', [('회피_01', '완주', _log_pass(), {}),
                                                    ('회피_02', 'no-data (60s 로그 정체)', None, None)])
    _write_arm(arm / '회피', '.', '20261002_110000', [('회피_02', 'blocked', _log_stop(), {})])
    runs = pr.load_arm('WON', arm, scen)
    got = {r['name']: (r['status'], r['dir'].name, r['log'] is not None) for r in runs}
    assert got == {'회피_01': ('완주', '20261002_100000', True),
                   '회피_02': ('blocked', '20261002_110000', True)}
    out = tmp_path / 'out'
    assert pr.main(['--arm', f'WON={arm}', '--scenarios', str(scen), '--out', str(out), '--no-figures',
                    '--arms-yaml', str(ROOT / 'config' / 'arms' / 'arms.yaml')]) == 0
    t = _rows(out / 'table2.csv')[0]
    assert (t['arm'], t['label'], t['n_valid'], t['completed'], t['inf_stop']) == \
        ('WON', '초기 구현(won)', '2', '1', '1')
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['warnings'] == [] and meta['arms']['WON']['label_en'] == 'Initial implementation (won)'
    # 꼬리표 없는 런(다른 브랜치에서 --config 없이 돈 결과)은 그렇다고 적힌다
    s = meta['arms']['WON']['settings']
    assert s['untagged_runs'] == 2 and s['runs_without_git'] == 2 and s['overlay'] == [] and s['code'] == []
    assert meta['scenario_manifest'] is None and any('MANIFEST' in w for w in meta['repro_warnings'])


def test_meta_records_code_settings_and_scenario_set(tmp_path):
    """재현 정보: 미커밋 여부 · overlay 키 전체 · 기준 params 해시 · MANIFEST 해시."""
    import hashlib
    scen = tmp_path / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    for n in ('회피_01', '회피_02'):
        _scenario(scen, '회피', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0))
    files = sorted(p for p in scen.rglob('*') if p.is_file())
    man = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(scen)}\n' for p in files)
    (scen / 'MANIFEST.sha256').write_text(man, encoding='utf-8')
    arm = tmp_path / 'WON24'
    _write_arm(tmp_path, 'WON24', '회피', [('회피_01', '완주', _log_pass(), {}), ('회피_02', '완주', _log_pass(), {})])
    rp = arm / '회피' / 'report.json'
    rep = json.loads(rp.read_text(encoding='utf-8'))
    ov = {'batch.stall_end_s': 60.0, 'batch.blocked_end_s': 60.0}
    code = {'commit': 'a' * 40, 'commit_short': 'a' * 7, 'branch': 'paper_exp', 'dirty': False,
            'modified': [], 'untracked': ['build/']}
    for i, res in enumerate(rep):
        res.update({'config': 'config/arms/WON24.yaml', 'arm': 'WON24', 'arm_overlay': ov,
                    'arm_base_sha256': 'b' * 64, 'judge': {'stall_end_s': 60.0},
                    # 두 번째 런은 트리를 고친 뒤 이어 돌린 것
                    'git': dict(code, dirty=bool(i), modified=[' M tools/batch_run.py'] if i else [])})
    rp.write_text(json.dumps(rep, ensure_ascii=False), encoding='utf-8')
    out = tmp_path / 'out'
    argv = ['--arm', f'WON24={arm}', '--scenarios', str(scen), '--out', str(out), '--no-figures',
            '--arms-yaml', str(tmp_path / 'no_arms.yaml')]
    assert pr.main(argv) == 0
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    now = pr.batch_run.git_state(ROOT)
    assert meta['git']['commit'] == now['commit'] and meta['commit'] == now['commit_short']
    assert set(meta['git']) == {'commit', 'commit_short', 'branch', 'dirty', 'modified', 'untracked'}
    assert meta['base_params']['sha256'] == hashlib.sha256((ROOT / 'config' / 'params.yaml').read_bytes()).hexdigest()
    m = meta['scenario_manifest']
    assert m['sha256'] == hashlib.sha256(man.encode()).hexdigest() and m['files'] == len(files) and m['mismatch'] == []
    s = meta['arms']['WON24']['settings']
    assert s['overlay'] == [ov] and s['base_params_sha256'] == ['b' * 64] and s['untagged_runs'] == 0
    assert sorted((c['dirty'], c['runs']) for c in s['code']) == [(False, 1), (True, 1)]
    assert any('런마다 코드·설정이 다르다' in w for w in meta['repro_warnings'])
    assert any('미커밋 변경이 있는 채로 돈 런 1건' in w for w in meta['repro_warnings'])
    # 생성 뒤 시나리오 파일이 바뀌면 잡힌다
    files[0].write_bytes(files[0].read_bytes() + b'\n')
    assert pr.main(argv) == 0
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['scenario_manifest']['mismatch'] == [str(files[0].relative_to(scen))]
    assert any('MANIFEST 와 다르다' in w for w in meta['repro_warnings'])


def test_missing_object_at_expected_spot_is_flagged(tmp_path):
    """시나리오 파일이 다르면(그 좌표에 물체가 없으면) 경고로 남는다."""
    scen = tmp_path / 'scenarios'
    _scenario(scen, '회피', '회피_01', [{'kind': 'static_vehicle', 'name': 'StoppedCar01',
                                        'route_s': OBS_X, 'lane': '자차로'}],
              _car_xml('StoppedCar01', OBS_X, 0.0))
    ticks = [_tick(i, 0.5 * i, 0.0, 10.0, []) for i in range(320)]      # 물체 없이 지나간다
    _write_arm(tmp_path, 'WON', '회피', [('회피_01', '완주', ticks, {})])
    out = tmp_path / 'out'
    pr.main(['--arm', f'WON={tmp_path / "WON"}', '--scenarios', str(scen), '--out', str(out), '--no-figures'])
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert any('시나리오 파일 불일치' in w for w in meta['warnings'])


def test_exclude_drops_scenario_from_every_arm(tmp_path):
    scen = tmp_path / 'scenarios'
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    for n in ('회피_01', '회피_02'):
        _scenario(scen, '회피', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0))
    (scen / 'batch_회피.json').write_text(json.dumps([{'name': '회피_01'}, {'name': '회피_02'}]))
    for arm in ('P', 'Q'):
        _write_arm(tmp_path, arm, '회피', [('회피_01', '완주', _log_pass(), {}),
                                          ('회피_02', 'blocked', _log_stop(), {})])
    out = tmp_path / 'out'
    pr.main(['--arm', f'P={tmp_path / "P"}', '--arm', f'Q={tmp_path / "Q"}', '--scenarios', str(scen),
             '--out', str(out), '--no-figures', '--exclude', '회피_02', '--ref', 'P'])
    t = {r['arm']: r for r in _rows(out / 'table2.csv')}
    assert all((t[x]['n_valid'], t[x]['n_expected'], t[x]['inf_stop']) == ('1', '1', '0') for x in 'PQ')
    assert [r['scenario'] for r in _rows(out / 'per_encounter.csv')] == ['회피_01']
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    assert meta['excluded_scenarios'] == ['회피_02'] and meta['excluded_runs'] == 2


# ── 신호 대기 줄 · 일반 주행 · 조건별 · 자동 제외 ────────────────────────
def _queue_log(shift_at=None, release=True, stop_behind=True):
    """정지선 앞 대기 차량(id 7) 뒤로 접근. release 면 300틱째 풀린다."""
    out = []
    for i in range(420):
        qv = 6.0 if (release and i >= 300) else 0.0
        qx = 120.0 + (0.3 * (i - 300) if release and i >= 300 else 0.0)
        ex = min(0.5 * i, 108.0) if stop_behind or i < 300 else 0.5 * i
        if release and i >= 320:
            ex = 108.0 + 0.3 * (i - 320)
        ev = 0.0 if 216 <= i < 320 else 6.0
        av = None
        if shift_at is not None and i == shift_at:
            av = {'state': 'PREEMPT', 'shift': 'left', 'blocker': 7, 'span': [900, 1300], 'span_new': [900, 1300]}
        out.append(_tick(i, ex, 0.0, ev, [(7, qx, 0.0, qv, 4.4, 1.8, 'vehicle')],
                         avoid=av, light=[5, 1 if i < 300 else 3], v_target=ev, dsl=14.0))
    return out


@pytest.fixture(scope='module')
def report2(tmp_path_factory):
    root = tmp_path_factory.mktemp('paper2')
    scen = root / 'scenarios'
    q = [{'kind': 'signal_queue', 'count': 1, 'queue': ['QueueCar01'], 'green_at_s': 15.0, 'stop_s': 124.0}]
    for n in ('줄_01', '줄_02', '줄_03'):
        _scenario(scen, '줄', n, q, _car_xml('QueueCar01', 120.0, 0.0))
    _scenario(scen, '주행', '주행_01', [{'kind': 'none'}], '')
    _scenario(scen, '주행', '주행_02', [{'kind': 'none'}], '')
    # 2026-10-03 규칙: 길이 차 ≥ 5 % 면 뺀다 (실경로 01_기본 처럼 2470 vs 6531 m)
    (scen / '주행' / '주행_02.placement.json').write_text(json.dumps(
        {'route_quality': {'builder_length_m': 2470.4, 'builder_off_lane': [],
                           'other_builder': {'name': 'won', 'rc': 0, 'same_lanes': False, 'off_lane': [],
                                             'length_m': 6531.4}}}), encoding='utf-8')
    ev = [{'kind': 'static_vehicle', 'name': 'StoppedCar01', 'route_s': OBS_X, 'lane': '자차로'}]
    for n, gap in (('거리_01', 8), ('거리_02', 30)):
        _scenario(scen, '거리', n, ev, _car_xml('StoppedCar01', OBS_X, 0.0))
        y = scen / '거리' / f'{n}.yaml'
        d = yaml.safe_load(y.read_text(encoding='utf-8'))
        d['axes'] = {'거리': gap, 'route': ['직진', 1, ''], 'event': ['static_vehicle']}
        y.write_text(yaml.safe_dump(d, allow_unicode=True), encoding='utf-8')
    drive = [_tick(i, 0.5 * i, 0.0, 10.0, []) for i in range(100)]
    for arm, shift in (('P', None), ('Q', 150)):
        _write_arm(root, arm, '줄', [('줄_01', '완주', _queue_log(shift_at=shift), {}),
                                    ('줄_02', 'blocked', _queue_log(release=False), {}),
                                    ('줄_03', '완주', _queue_log(), {})])
        _write_arm(root, arm, '주행', [('주행_01', '완주' if arm == 'P' else 'timeout', drive, {}),
                                      ('주행_02', '완주', drive, {})])
        _write_arm(root, arm, '거리', [('거리_01', 'blocked' if arm == 'Q' else '완주',
                                       _log_stop() if arm == 'Q' else _log_pass(), {}),
                                      ('거리_02', '완주', _log_pass(), {})])
    for arm in ('P', 'Q'):                               # 일반 주행의 시간·감점은 report.json 에서
        rp = root / arm / '주행' / 'report.json'
        rr = json.loads(rp.read_text(encoding='utf-8'))
        for r in rr:
            r.update(time_s=77.0, deduction=-6)
        rp.write_text(json.dumps(rr, ensure_ascii=False), encoding='utf-8')
    out = root / 'out'
    cfgp = root / 'arms.yaml'
    cfgp.write_text(yaml.safe_dump({'report': {'reference': 'P', 'condition_figures': [
        {'name': 'fig_near_stop_distance', 'types': ['거리'], 'axis': '거리', 'xlabel': 'gap'}]}},
        allow_unicode=True), encoding='utf-8')
    assert pr.main(['--arm', f'P={root / "P"}', '--arm', f'Q={root / "Q"}', '--scenarios', str(scen),
                    '--out', str(out), '--arms-yaml', str(cfgp)]) == 0
    return out


def test_queue_success_false_positive_and_not_released(report2):
    q = {(r['arm'], r['scenario']): r for r in _rows(report2 / 'queue_fp.csv')}
    assert q[('P', '줄_01')]['success'] == 'True' and q[('P', '줄_01')]['shifts_on_queue'] == '0'
    fp = q[('Q', '줄_01')]                                   # 적색 동안 대기 차량에 시프트 = 오탐
    assert (fp['success'], fp['why'], fp['shifts_on_queue'], fp['shift_tick']) == \
        ('False', 'false_positive_shift', '1', '150')
    assert fp['light_state_at_shift'] == 'red' and float(fp['shift_t_s']) < float(fp['t_release_s'])
    # 대기 차량이 끝까지 안 풀려 생긴 정지는 무한 정지 실패로 세지 않고 따로 분류한다
    nr = q[('P', '줄_02')]
    assert (nr['success'], nr['why'], nr['t_release_s']) == ('', 'queue_not_released', '')
    bt = {(r['type'], r['arm']): r for r in _rows(report2 / 'by_type.csv')}
    assert (bt[('줄', 'P')]['success'], bt[('줄', 'P')]['n_judged'], bt[('줄', 'P')]['n_excluded']) == ('2', '2', '1')
    assert (bt[('줄', 'Q')]['success'], bt[('줄', 'Q')]['false_positive_runs']) == ('1', '1')
    assert 'queue_not_released=1' in bt[('줄', 'P')]['excluded_reasons']


def test_plain_driving_and_auto_exclusion(report2):
    meta = json.loads((report2 / 'meta.json').read_text(encoding='utf-8'))
    # 구현이 둘 이상이면 두 빌더의 경로가 다른 시나리오를 자동으로 뺀다
    assert meta['auto_excluded_builders_differ'] == ['주행_02'] and meta['excluded_scenarios'] == ['주행_02']
    pl = {(r['arm'], r['scenario']): r for r in _rows(report2 / 'plain.csv')}
    assert set(pl) == {('P', '주행_01'), ('Q', '주행_01')}
    assert (pl[('P', '주행_01')]['success'], pl[('P', '주행_01')]['time_s'], pl[('P', '주행_01')]['deduction']) == \
        ('True', '77.0', '-6')
    assert pl[('Q', '주행_01')]['success'] == 'False'
    bt = {(r['type'], r['arm']): r for r in _rows(report2 / 'by_type.csv')}
    assert (bt[('주행', 'P')]['class'], bt[('주행', 'P')]['mean_time_s'], bt[('주행', 'P')]['mean_deduction']) == \
        ('plain', '77.0', '-6.0')


def test_by_condition_and_condition_figure(report2):
    bc = {(r['axis'], r['value'], r['arm']): r for r in _rows(report2 / 'by_condition.csv') if r['type'] == '거리'}
    assert (bc[('거리', '8', 'P')]['success_pct'], bc[('거리', '8', 'Q')]['success_pct']) == ('100.0', '0.0')
    assert bc[('거리', '30', 'Q')]['success_pct'] == '100.0'
    f = report2 / 'figures'
    assert (f / 'fig_near_stop_distance.csv').exists() and (f / 'fig_near_stop_distance.png').stat().st_size > 1000
    p = {r['unit']: r for r in _rows(report2 / 'paired.csv') if r['other'] == 'Q'}
    assert p['scenario(성공, 회피)']['ref_only_success'] == '1'
    assert p['scenario(성공, 오탐·신호 대기 줄)']['n_pairs'] == '2'       # 줄_02 는 양쪽 다 판정 제외


def test_builders_differ_rule():
    """2026-10-03 결정: 장애물 차로를 한쪽이라도 안 지나거나 길이 차 ≥ 5 % 일 때만 비교에서 뺀다."""
    def sc(off_this=(), off_other=(), lt=600.0, lo=605.0, rc=0):
        return {'builders': {'other': 'won', 'other_rc': rc, 'off_lane_this': list(off_this),
                             'off_lane_other': list(off_other), 'len_this_m': lt, 'len_other_m': lo}}
    assert pr.builders_differ(sc(), 0.05) is None                    # 차로열이 조금 달라도 같은 길
    assert pr.builders_differ(sc(lo=630.0), 0.05) is not None         # 5.0 %
    assert pr.builders_differ(sc(lo=629.0), 0.05) is None
    assert pr.builders_differ(sc(lt=2470.4, lo=6531.4), 0.05) is not None
    assert pr.builders_differ(sc(off_other=[(1, 0, -1)]), 0.05) is not None
    assert pr.builders_differ(sc(off_this=[(1, 0, -1)]), 0.05) is not None
    assert pr.builders_differ(sc(rc=1), 0.05) is not None
    assert pr.builders_differ({'builders': None}, 0.05) is None and pr.builders_differ(None, 0.05) is None
