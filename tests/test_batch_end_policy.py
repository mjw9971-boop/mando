"""
배치 조기 종료 정책(--no / --timeout)과 시나리오 타임아웃 상수 (2026-09-07).

배경: 낮에는 사람이 보면서 끝까지 돌리고 싶고, 밤 무인 배치는 막히면 끊고 다음으로
넘어가야 한다. 시간대 자동 전환은 하지 않는다 — 자정을 넘기면 한 배치 안에서
규칙이 갈려 런끼리 비교가 안 된다. 그래서 플래그로 사람이 정한다.

타임아웃: 예전 상수는 평균 27 km/h · 배수 1.8 · 오버헤드 90 s 였는데 9/6 실주행
평균이 10.2 km/h(신호 대기 포함)라 3 km 경로가 810 s 로 잘려 완주 전에 죽었다.
"""
import io
import contextlib
import json
import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import batch_run as BR                                          # noqa: E402
import gen_scenarios as GS                                      # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402

CFG = load_params_yaml()


def _dry(argv):
    """--dry-run 으로 main 을 태우고 stdout 을 돌려준다."""
    sc = [{'name': 'x', 'vtd_xml_path': '/tmp/x.xml',
           'route_csv': 'waypoints.csv', 'timeout_s': 300}]
    p = pathlib.Path(tempfile.mkdtemp()) / 'sc.json'
    p.write_text(json.dumps(sc), encoding='utf-8')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            BR.main([str(p), '--dry-run'] + argv)
        except SystemExit:
            pass
    return buf.getvalue()


def _policy_line(out):
    return next(l for l in out.splitlines() if l.startswith('종료 정책'))


def test_default_is_early_stop():
    assert '--timeout' in _policy_line(_dry([]))


def test_no_and_no_timeout_are_the_same():
    a = _policy_line(_dry(['--no']))
    b = _policy_line(_dry(['--no-timeout']))
    assert a == b
    assert '--no' in a and '조기 종료 없음' in a


def test_timeout_flag_is_explicit_default():
    assert _policy_line(_dry(['--timeout'])) == _policy_line(_dry([]))


def test_policy_recorded_on_runner():
    class A:
        no_early = True
        dry_run = True
        host = 'x'
    r = BR.Runner.__new__(BR.Runner)
    r.end_policy = 'no_timeout' if A.no_early else 'default'
    assert r.end_policy == 'no_timeout'


# ── 타임아웃 상수 ────────────────────────────────────────────────────────
def test_timeout_params_present():
    b = CFG['batch']
    assert float(b['timeout_avg_kph']) == 12.0
    assert float(b['timeout_factor']) == 1.5
    assert float(b['timeout_overhead_s']) == 120.0
    assert float(b['timeout_min_s']) == 300.0


def test_timeout_for_3km_is_about_1470s():
    GS._TIMEOUT_CFG = None
    GS.timeout_cfg(reload=True)
    assert abs(GS.timeout_for(3000.0) - 1470) <= 5


def test_est_seconds_does_not_apply_factor():
    """예상시간은 실제로 걸릴 시간이다 — 타임아웃 배수를 곱하지 않는다."""
    GS.timeout_cfg(reload=True)
    avg, factor, overhead, _min = GS.timeout_cfg()
    assert factor > 1.0
    assert abs(GS.est_seconds(3000.0) - (3000.0 / avg + GS.OVERHEAD_S)) < 1e-6
    assert GS.est_seconds(3000.0) < GS.timeout_for(3000.0)


def test_timeout_falls_back_to_old_constants(monkeypatch):
    """params 키가 없으면 옛 값(27 km/h · 1.8 · 90 · 180)으로 떨어진다."""
    import vtd_adapter.config as C
    monkeypatch.setattr(C, 'load_params_yaml', lambda *a, **k: {'batch': {}})
    GS._TIMEOUT_CFG = None
    avg, factor, overhead, min_s = GS.timeout_cfg(reload=True)
    assert abs(avg - 27.0 / 3.6) < 1e-9
    assert (factor, overhead, min_s) == (1.8, 90.0, 180.0)
    assert GS.timeout_for(3000.0) == max(180, int(3000.0 / avg * 1.8 + 90))
    GS._TIMEOUT_CFG = None
    GS.timeout_cfg(reload=True)


def test_batch_all_has_real_route_scenario():
    """실경로 CSV 로 만든 시나리오가 배치 목록에 들어가고 timeout 이 붙는다."""
    p = ROOT / 'scenarios' / 'batch_all.json'
    if not p.exists():
        pytest.skip('scenarios/ 생성물 없음')
    items = json.loads(p.read_text(encoding='utf-8'))
    real = [i for i in items if i['name'].startswith('실경로_')]
    assert real, '실경로_* 시나리오가 batch_all.json 에 없다'
    for i in real:
        assert i['timeout_s'] >= float(CFG['batch']['timeout_min_s'])
        assert (ROOT / i['route_csv']).exists()
