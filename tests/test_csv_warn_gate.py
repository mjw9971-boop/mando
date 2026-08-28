"""
--csv 로 주행을 시작할 때 build_route 경고를 무시하지 않는다.

build_route 의 경고는 전부 "이 경로로 달리면 문제가 생긴다" 류다 — 경유점을
스치지 않음 / junction 미경유 / 교차로 내부 차선변경 / 차선변경 창 부족 /
회전 불가 기하. 예전에는 rc=1 을 "진행 가능" 으로 통과시켜서, 대회날 급하게
`run_agent.py --csv` 한 방으로 돌리면 리포트를 아무도 안 보고 출발할 수 있었다
(README 체크리스트의 눈검사가 건너뛰어진다).

  rc=0 (경고 없음)          → 진행
  rc=1 (경고) + 플래그 없음 → 중단
  rc=1 (경고) + 허용 플래그 → 진행
  rc=2 (실패)               → 중단
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_agent  # noqa: E402


@pytest.fixture()
def fake_build(monkeypatch, tmp_path):
    """build_route.py 를 부르지 않고 종료코드만 흉내낸다."""
    def make(rc: int, create: bool = True):
        def fake_run(cmd, **kw):
            if create:
                pathlib.Path(cmd[cmd.index('-o') + 1]).write_bytes(b'x')
            return subprocess.CompletedProcess(cmd, rc)
        monkeypatch.setattr(run_agent.subprocess, 'run', fake_run)
        return str(tmp_path / 'wp.csv')
    return make


def test_clean_build_proceeds(fake_build, tmp_path):
    csv = fake_build(0)
    out = run_agent.build_route_from_csv(csv, 'g.pkl', str(tmp_path))
    assert pathlib.Path(out).exists()


def test_warning_aborts(fake_build, tmp_path):
    csv = fake_build(1)
    with pytest.raises(SystemExit) as e:
        run_agent.build_route_from_csv(csv, 'g.pkl', str(tmp_path))
    assert '--allow-route-warnings' in str(e.value)


def test_warning_can_be_overridden(fake_build, tmp_path):
    csv = fake_build(1)
    out = run_agent.build_route_from_csv(csv, 'g.pkl', str(tmp_path),
                                         allow_warnings=True)
    assert pathlib.Path(out).exists()


def test_hard_failure_aborts_even_with_flag(fake_build, tmp_path):
    csv = fake_build(2, create=False)
    with pytest.raises(SystemExit) as e:
        run_agent.build_route_from_csv(csv, 'g.pkl', str(tmp_path),
                                       allow_warnings=True)
    assert 'build_route 실패' in str(e.value)


def test_flag_is_wired_to_parser():
    """--allow-route-warnings 가 실제로 빌드 호출에 전달되는지 (기본은 False)."""
    a = run_agent.build_parser().parse_args(['--csv', 'x.csv'])
    assert a.allow_route_warnings is False
    a = run_agent.build_parser().parse_args(['--csv', 'x.csv', '--allow-route-warnings'])
    assert a.allow_route_warnings is True
