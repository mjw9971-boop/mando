"""논문 실험 도구 — 구현별 설정 사본(make_arms) · batch_run --config/--out-dir/--resume · 실행 계획(paper_run).

지키는 것 (2026-10-01):
  · 설정 사본은 config/arms/ 의 overlay 로만 만든다. 2026-10-02 부터 **주행 파라미터는
    사본에서도 바꾸지 않는다** — arms.yaml allowed_sections(batch) 밖은 거부한다.
  · overlay 의 키는 기준 params 에
    **이미 있는 키**여야 하고(오타가 새 키로 조용히 들어가면 스위치가 안 먹은 채 돈다),
    사본이 기준과 다른 키는 overlay 로 바뀐 키와 정확히 같다.
  · batch_run 은 --config 를 주면 종료 판정(batch.*)과 run_agent 가 같은 파일을 읽고,
    안 주면 종전과 같다 (launch 명령에 --config 가 없고 결과에 꼬리표가 없다).
  · --out-dir + --resume: 런마다 report.json 이 갱신되고, 끝난 런은 건너뛰며,
    기반 실패(SCP·no-data 등)는 다시 돈다.

사본이 params 와 **최신인지**는 여기서 검사하지 않는다 — 제어기 파트가 params 를
튜닝할 때마다 이 테스트가 깨지면 안 된다. 그 검사는 paper_run 이 실행 직전에 한다.
"""
import json
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import batch_run as br                                          # noqa: E402
import make_arms as ma                                          # noqa: E402
import paper_run as pr                                          # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402


# ── make_arms ────────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def defs():
    return ma.load_defs()


@pytest.fixture(scope='module')
def built(defs):
    return {name: ma.build(name, defs) for name in defs['arms']}


def test_overlay_keys_exist_in_base_and_only_they_change(defs, built):
    base = ma.flatten(load_params_yaml())
    for name, (text, arm) in built.items():
        cfg = yaml.safe_load(text)
        assert cfg.pop('arm') == arm
        after = ma.flatten(cfg)
        assert set(arm['overlay']) <= set(base), name            # 새 키를 만들지 않는다
        diff = {k for k in base if base[k] != after[k]}
        assert set(after) == set(base) and diff == set(arm['changed']), name
        for k, (old, new) in arm['changed'].items():
            assert base[k] == old and after[k] == new and arm['overlay'][k] == new


A2_KEYS = {'ctrl24.enable', 'speed.margin_kph', 'speed.approach_decel_mps2', 'overtake.a_lat_max'}


def test_copies_change_judge_values_and_a2_keys_only(defs, built):
    """WON24 사본은 배치 판정값만 바뀐다 (주행 파라미터 그대로). A2(초기 구현, 2026-10-04)는
    거기에 ctrl24.enable=false 와 won 의 주행 값 세 개뿐이다 — 그 밖의 키는 기준과 같다."""
    assert defs['allowed_sections'] == ['batch'] and set(defs['allowed_keys']) == A2_KEYS
    base = load_params_yaml()
    fb = ma.flatten(base)
    for name, (text, arm) in built.items():
        cfg = yaml.safe_load(text)
        cfg.pop('arm')
        fc = ma.flatten(cfg)
        changed = {k for k in fb if fb[k] != fc[k]}
        allowed = {k for k in fb if k.startswith('batch.')} | (A2_KEYS if name == 'A2' else set())
        assert changed <= allowed, f'{name}: {sorted(changed - allowed)}'
        assert cfg['batch']['stall_end_s'] == cfg['batch']['blocked_end_s'] == 60.0
        assert cfg['batch']['no_progress_end_s'] == base['batch']['no_progress_end_s']
        assert arm['label_ko'] and arm['label_en'] and arm['builder'] == 'paper_exp'
    assert built['WON24'][1]['label_ko'] == '최종 구현(won_24)'
    assert {k for k in built['WON24'][1]['overlay']} <= {k for k in fb if k.startswith('batch.')}
    a2 = yaml.safe_load(built['A2'][0])
    # won(origin/won 9b05f1f) 의 값 — kr 경로가 읽는 주행 키 세 개 (리플레이 근거: docs "A2")
    assert a2['ctrl24']['enable'] is False and a2['speed']['margin_kph'] == 5.0
    assert a2['speed']['approach_decel_mps2'] == 2.0 and a2['overtake']['a_lat_max'] == 1.5
    assert built['A2'][1]['label_ko'] == '초기 구현(won)'


def test_allowed_keys_open_single_keys_not_sections():
    cfg = {'ctrl24': {'enable': True, 'avoid_enable': True}, 'batch': {'stall_end_s': 1.0}}
    keys = ['ctrl24.enable']
    ma.apply_overlay(cfg, {'ctrl24': {'enable': False}}, 'x', allowed=['batch'], allowed_keys=keys)
    assert cfg['ctrl24']['enable'] is False
    with pytest.raises(ma.ArmError):                             # 같은 섹션의 다른 키는 여전히 거부
        ma.apply_overlay(cfg, {'ctrl24': {'avoid_enable': False}}, 'x', allowed=['batch'], allowed_keys=keys)
    assert cfg['ctrl24']['avoid_enable'] is True


def test_overlay_outside_allowed_sections_is_rejected():
    cfg = {'ctrl24': {'avoid_enable': True}, 'batch': {'stall_end_s': 1.0}}
    with pytest.raises(ma.ArmError):
        ma.apply_overlay(cfg, {'ctrl24': {'avoid_enable': False}}, 'x.overlay.yaml', allowed=['batch'])
    assert cfg['ctrl24']['avoid_enable'] is True                 # 거부된 overlay 는 값을 건드리지 않는다
    ma.apply_overlay(cfg, {'batch': {'stall_end_s': 60.0}}, 'x.overlay.yaml', allowed=['batch'])
    assert cfg['batch']['stall_end_s'] == 60.0


def test_unknown_overlay_key_is_rejected():
    cfg = {'ctrl24': {'avoid_enable': True}}
    with pytest.raises(ma.ArmError):
        ma.apply_overlay(cfg, {'ctrl24': {'avoid_enabel': False}}, 'x.overlay.yaml')
    with pytest.raises(ma.ArmError):
        ma.apply_overlay(cfg, {'ctrl25': {'avoid_enable': False}}, 'x.overlay.yaml')
    ma.apply_overlay(cfg, {'ctrl24': {'avoid_enable': False}}, 'x.overlay.yaml')
    assert cfg == {'ctrl24': {'avoid_enable': False}}


# ── batch_run --config / --out-dir / --resume ────────────────────────────
def _scenarios(tmp_path, n=3):
    items = []
    for i in range(n):
        xml = tmp_path / f's{i}.xml'
        xml.write_text('<x/>')
        items.append({'name': f's{i}', 'vtd_xml_path': str(xml),
                      'route_csv': str(tmp_path / f's{i}.csv'), 'timeout_s': 300})
    lst = tmp_path / 'batch.json'
    lst.write_text(json.dumps(items), encoding='utf-8')
    return lst


def _runner(argv, statuses, calls):
    a = br.build_parser().parse_args(argv)
    r = br.Runner(a)

    def fake_run_one(sc):
        calls.append(sc['name'])
        return {'name': sc['name'], 'status': statuses[sc['name']], 'log': None}
    r.run_one = fake_run_one
    return r


def test_defaults_are_the_previous_behaviour(tmp_path, capsys):
    a = br.build_parser().parse_args([str(_scenarios(tmp_path)), '--dry-run', '--pause-s', '0'])
    assert (a.config, a.out_dir, a.resume) == (None, None, False)
    r = br.Runner(a)
    assert r.cfg_path is None and r.arm_info() == {}
    assert r.out_dir.parent == ROOT / 'logs' / 'batch'
    r.run()
    out = capsys.readouterr().out
    assert '--config' not in out and '설정:' not in out
    assert all('arm' not in res and 'config' not in res and 'git' not in res for res in r.results)


def test_config_reaches_judge_and_agent(tmp_path, capsys):
    cfg = load_params_yaml()
    cfg['batch']['stall_end_s'] = cfg['batch']['blocked_end_s'] = 61.0
    cfg['arm'] = {'name': 'AX', 'label_ko': '시험', 'overlay': {'batch.stall_end_s': 61.0},
                  'base_sha256': 'abc'}
    cpath = tmp_path / 'AX.yaml'
    cpath.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    a = br.build_parser().parse_args([str(_scenarios(tmp_path)), '--dry-run', '--pause-s', '0',
                                      '--config', str(cpath)])
    r = br.Runner(a)
    assert r.batch_cfg['stall_end_s'] == r.batch_cfg['blocked_end_s'] == 61.0
    r.run()
    out = capsys.readouterr().out
    assert f'--config {cpath}' in out                       # run_agent 에 같은 파일
    assert 'stall(계획은 진행인데 못 감) 61 s' in out          # 자기 판정도 그 파일
    tag = r.arm_info()
    assert tag['arm'] == 'AX' and tag['arm_overlay'] == {'batch.stall_end_s': 61.0}
    assert tag['judge']['stall_end_s'] == 61.0 and tag['judge']['blocked_end_s'] == 61.0
    assert all(res['arm'] == 'AX' and res['config'] == str(cpath) for res in r.results)
    # 런마다 코드 상태(커밋 해시 · 미커밋 여부)가 붙는다 — 해시만으로는 돌린 코드를 못 가린다.
    now = br.git_state()
    for res in r.results:
        assert res['git']['commit'] == now['commit'] and len(res['git']['commit']) == 40
        assert isinstance(res['git']['dirty'], bool) and res['git']['branch'] == now['branch']


def test_git_state_separates_tracked_changes_from_untracked(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)
    git('init', '-q')
    git('config', 'user.email', 't@t'), git('config', 'user.name', 't')
    (tmp_path / 'a.txt').write_text('1')
    git('add', 'a.txt'), git('commit', '-qm', 'x')
    st = br.git_state(tmp_path)
    assert st['dirty'] is False and st['modified'] == [] and st['untracked'] == []
    (tmp_path / 'build').mkdir()
    (tmp_path / 'build' / 'o').write_text('')                # 추적 안 된 산출물만으로는 dirty 가 아니다
    st = br.git_state(tmp_path)
    assert st['dirty'] is False and st['untracked'] == ['build/']
    (tmp_path / 'a.txt').write_text('2')
    st = br.git_state(tmp_path)
    assert st['dirty'] is True and st['modified'] == [' M a.txt'] and st['commit_short'] == st['commit'][:7]
    # 저장소가 아니면 배치를 막지 않고 전부 None
    assert br.git_state(tmp_path / 'build' / '없는폴더')['commit'] is None


def test_resume_needs_out_dir(tmp_path):
    with pytest.raises(SystemExit):
        br.main([str(_scenarios(tmp_path)), '--resume'])


def test_out_dir_saves_each_run_and_resume_skips_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(br.time, 'sleep', lambda _s: None)
    lst = _scenarios(tmp_path)
    out = tmp_path / 'paper' / 'A1' / '유형'
    argv = [str(lst), '--out-dir', str(out), '--resume', '--pause-s', '0']
    calls: list = []
    # 1차: s0 완주, s1 기반 실패, s2 blocked
    r = _runner(argv, {'s0': '완주', 's1': 'SCP 실패: boom', 's2': 'blocked'}, calls)
    seen = []
    orig = r.save_partial

    def spy():
        orig()
        seen.append(len(json.loads((out / 'report.json').read_text(encoding='utf-8'))))
    r.save_partial = spy
    r.run()
    assert calls == ['s0', 's1', 's2'] and seen == [1, 2, 3]   # 런마다 report.json 갱신
    # 2차: 끝난 런(완주·blocked)은 건너뛰고 기반 실패만 다시 돈다
    calls2: list = []
    r2 = _runner(argv, {'s0': 'X', 's1': 'stall', 's2': 'X'}, calls2)
    r2.run()
    assert calls2 == ['s1']
    rep = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    assert [(x['name'], x['status']) for x in rep] == [('s0', '완주'), ('s1', 'stall'), ('s2', 'blocked')]
    # 3차: 전부 끝났다 — 아무것도 돌지 않는다
    calls3: list = []
    _runner(argv, {}, calls3).run()
    assert calls3 == []


def test_done_status_is_real_outcomes_only():
    assert set(br.Runner.DONE_STATUS) == {'완주', 'stall', 'blocked', 'no_progress', 'timeout'}


# ── paper_run 계획 ───────────────────────────────────────────────────────
def test_type_of():
    assert pr.type_of('batch_추월집중_논문.json') == '추월집중_논문'
    assert pr.type_of('scenarios/batch_급정거집중.json') == '급정거집중'


def test_plan_covers_every_list_with_one_arm(defs):
    """계획 = 목록 파일 × (WON24, A2). 개수는 목록 파일이 정한다 (코드에 적지 않는다)."""
    sc = ROOT / defs['scenario_dir']
    lists = [ln for block in defs['plan'] for ln in block['lists']]
    if not all((sc / ln).exists() for ln in lists):
        pytest.skip('scenarios/ 생성물 없음')
    plan = pr.build_plan(defs)
    arms = defs['plan'][0]['arms']
    assert arms == ['WON24', 'A2']
    assert [p['arm'] for p in plan] == [a for a in arms for _ in lists]
    assert [p['type'] for p in plan] == [pr.type_of(ln) for _ in arms for ln in lists]
    for p in plan:
        ln = f'batch_{p["type"]}.json'
        assert len(p['items']) == len(json.loads((sc / ln).read_text(encoding='utf-8')))
        assert p['out_dir'] == ROOT / defs['out_dir'] / p['arm'] / p['type']
        assert p['config'] == ma.ARMS_DIR / f'{p["arm"]}.yaml'
    only = pr.build_plan(defs, arms=['WON24'], types=[pr.type_of(lists[-1])])
    assert [(p['arm'], p['type']) for p in only] == [('WON24', pr.type_of(lists[-1]))]
