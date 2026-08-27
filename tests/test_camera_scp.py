"""
batch_run IG 팔로워 뷰 카메라 (표시 전용 — GT·제어·채점 무관).

  · build_camera_scp: params camera.* → SCP <Camera> 문자열 (단일 출처, 하드코딩 금지)
  · run_one 의 SCP 순서: Load → Init → Camera(1회) → Start
    (mock_vtd 는 9910 전용이라 SCP 를 받지 않는다 — scp_client 스텁으로
     호출 순서를 확인한다. 카메라 실패는 배치를 멈추지 않는다)
"""
import copy
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'

import batch_run                                   # noqa: E402 (conftest 가 tools 경로 추가)
from vtd_adapter.config import load_params_yaml    # noqa: E402

PRESET = ('<Camera name="followCam" showOwner="true">'
          '<PosRelative player="Ego" dx="-6.50" dy="0.00" dz="2.60"/>'
          '<ViewRelative dh="0.0000" dp="0.0000" dr="0.0000"/>'
          '<Set/></Camera>')


@pytest.fixture()
def cfg():
    return copy.deepcopy(load_params_yaml())


# ── 단위: build_camera_scp ────────────────────────────────────────────────
def test_preset_builds_exact_string(cfg):
    assert batch_run.build_camera_scp(cfg) == PRESET


def test_disabled_returns_none(cfg):
    cfg['camera']['enabled'] = False
    assert batch_run.build_camera_scp(cfg) is None


def test_show_owner_false(cfg):
    cfg['camera']['show_owner'] = False
    assert 'showOwner="false"' in batch_run.build_camera_scp(cfg)


def test_missing_section_raises(cfg):
    del cfg['camera']
    with pytest.raises(KeyError):
        batch_run.build_camera_scp(cfg)


def test_missing_key_raises(cfg):
    del cfg['camera']['dz']
    with pytest.raises(KeyError):
        batch_run.build_camera_scp(cfg)


# ── 통합: run_one 의 SCP 호출 순서 (스텁) ─────────────────────────────────
class ScpStub:
    """ScpClient 대역 — 호출 순서만 기록한다."""

    def __init__(self, *_a, **_k):
        self.calls = []

    def connect(self):
        return self

    def load_scenario(self, path):
        self.calls.append('load')

    def init(self):
        self.calls.append('init')

    def send(self, payload, receiver='any'):
        self.calls.append(('camera', payload))

    def start(self, t_max=None):
        self.calls.append('start')

    def poll(self, seconds=1.0):
        return []

    def stop(self):
        self.calls.append('stop')

    def close(self):
        pass


class Args:
    host = '127.0.0.1'
    ssh = None
    settle_s = 0.0
    vtd_warmup_s = 0.0     # 9910 대기 즉시 포기 → SCP 순서까지만 진행하고 반환
    pause_s = 0.0
    dry_run = False
    scenarios = []


SCENARIOS = ['완주속도_01_속도전환3', '완주속도_02_좌회전2', '완주속도_03_연속교차로3']

needs_files = pytest.mark.skipif(
    not GRAPH.exists()
    or not all((ROOT / 'scenarios' / '완주속도' / f'{n}.csv').exists() for n in SCENARIOS),
    reason='lane_graph.pkl 또는 완주속도 시나리오 csv 없음')


@needs_files
def test_camera_order_per_scenario(tmp_path, monkeypatch):
    """완주속도 3개: 시나리오마다 Load → Init → Camera(정확히 1회) → Start,
    run 메타에 camera_sent=True 기록. 9910 이 없으니 그 직후 '무응답'으로
    끝나는 것까지가 기대 흐름이다 (SCP 순서는 이미 다 탄 뒤)."""
    created = []

    class RecordingStub(ScpStub):
        def __init__(self, *a, **k):
            super().__init__()
            created.append(self)

    monkeypatch.setattr(batch_run, 'ScpClient', RecordingStub)
    runner = batch_run.Runner(Args())
    runner.out_dir = tmp_path
    for name in SCENARIOS:
        res = runner.run_one({'name': name,
                              'vtd_xml_path': f'/x/{name}.xml',
                              'route_csv': f'scenarios/완주속도/{name}.csv', 'timeout_s': 60})
        assert res['status'].startswith('VTD 9910 무응답'), res
        assert res['camera_sent'] is True
    assert len(created) == len(SCENARIOS)
    for stub in created:
        kinds = [c[0] if isinstance(c, tuple) else c for c in stub.calls]
        assert kinds[:4] == ['load', 'init', 'camera', 'start'], kinds
        assert kinds.count('camera') == 1
        cam = next(c[1] for c in stub.calls if isinstance(c, tuple))
        assert cam == PRESET


def test_camera_failure_does_not_stop_batch(cfg):
    """send 가 소켓 오류를 내도 경고만 남고 False 로 기록된다."""
    class Boom(ScpStub):
        def send(self, payload, receiver='any'):
            raise OSError('socket down')

    runner = batch_run.Runner.__new__(batch_run.Runner)
    runner.cfg = cfg
    runner.scp = Boom()
    assert runner._apply_camera() is False
    cfg['camera']['enabled'] = False
    assert runner._apply_camera() is None
