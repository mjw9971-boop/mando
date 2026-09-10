"""배치용 params 오버레이 (`config/params_batch.yaml`, MANDO_PARAMS_OVERLAY).

**왜 있나** (2026-09-10): 실험 스위치를 `config/params.yaml` 에서 켜 두면
"기본값이 꺼져 있는지" 보는 검사가 통째로 깨진다. 2026-09-07 에 정리했던
바로 그 드리프트가 다시 났고(`pytest` 35 failed, 전부 `*_default_is_off`
계열), 그 상태로는 무엇이 새 실패인지 구분이 안 돼 회귀 판정이 불가능하다.

경계는 하나다: **오버레이는 명시적으로 요청할 때만 얹힌다.** pytest 는
환경변수를 안 걸므로 언제나 커밋본(전부 false)을 본다.
"""
import os
import pathlib
import sys
import textwrap

import pytest
import yaml

from vtd_adapter.config import (DEFAULT_OVERLAY, OVERLAY_ENV, ROOT,
                                load_params_yaml)

BATCH_YAML = ROOT / 'config' / 'params_batch.yaml'


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """환경변수는 매 검사 지운다 — 배치 셸에서 pytest 를 돌려도 같은 결과."""
    monkeypatch.delenv(OVERLAY_ENV, raising=False)


def write(tmp_path, text):
    p = tmp_path / 'over.yaml'
    p.write_text(textwrap.dedent(text), encoding='utf-8')
    return str(p)


# ── 기본은 "없음" ───────────────────────────────────────────────────────
def test_default_has_no_overlay():
    """이 파일이 지키는 핵심 — 아무것도 안 걸면 커밋본 그대로다."""
    cfg = load_params_yaml()
    raw = yaml.safe_load(BATCH_YAML.read_text(encoding='utf-8'))
    for sec, keys in raw.items():
        for k, v in keys.items():
            assert cfg[sec][k] != v or not isinstance(v, bool), (
                f'{sec}.{k} 가 커밋본에서 이미 오버레이 값이다')


def test_committed_params_keep_the_batch_switches_off():
    """커밋본은 전부 false 유지 (사용자 지시 2026-09-10)."""
    cfg = load_params_yaml()
    raw = yaml.safe_load(BATCH_YAML.read_text(encoding='utf-8'))
    on = [f'{s}.{k}' for s, ks in raw.items() for k, v in ks.items()
          if v is True and cfg[s][k] is True]
    assert on == []


def test_explicit_path_beats_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv(OVERLAY_ENV, '1')
    p = write(tmp_path, 'percep:\n  route_index_hops: 5\n')
    assert load_params_yaml(overlay=p)['percep']['route_index_hops'] == 5


def test_empty_string_overlay_disables_it(monkeypatch):
    monkeypatch.setenv(OVERLAY_ENV, '1')
    base = load_params_yaml(overlay='')
    assert base['avoid_map']['lane_map_avoid_enable'] is False


@pytest.mark.parametrize('raw', ['0', 'false', 'no', 'off', '', '  '])
def test_falsy_env_values_disable_it(monkeypatch, raw):
    monkeypatch.setenv(OVERLAY_ENV, raw)
    assert load_params_yaml()['avoid_map']['lane_map_avoid_enable'] is False


@pytest.mark.parametrize('raw', ['1', 'true', 'yes', 'on', 'TRUE'])
def test_truthy_env_values_pick_the_batch_file(monkeypatch, raw):
    monkeypatch.setenv(OVERLAY_ENV, raw)
    assert load_params_yaml()['avoid_map']['lane_map_avoid_enable'] is True
    assert DEFAULT_OVERLAY == BATCH_YAML


def test_env_can_name_another_file(tmp_path, monkeypatch):
    """실험 세트를 여러 벌 두는 통로."""
    monkeypatch.setenv(OVERLAY_ENV, write(tmp_path, 'percep:\n  route_index_hops: 4\n'))
    assert load_params_yaml()['percep']['route_index_hops'] == 4


# ── 오타 가드 ───────────────────────────────────────────────────────────
def test_unknown_key_dies(tmp_path):
    """조용히 새 키를 만들면 켠 줄 알았는데 아무 일도 안 일어난다."""
    p = write(tmp_path, 'avoid_map:\n  lane_map_avoid_enabel: true\n')
    with pytest.raises(KeyError, match='lane_map_avoid_enabel'):
        load_params_yaml(overlay=p)


def test_unknown_section_dies(tmp_path):
    p = write(tmp_path, 'avoid_mapp:\n  lane_map_avoid_enable: true\n')
    with pytest.raises(KeyError, match='avoid_mapp'):
        load_params_yaml(overlay=p)


def test_missing_overlay_file_dies(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params_yaml(overlay=str(tmp_path / 'nope.yaml'))


# ── 병합 규칙 ───────────────────────────────────────────────────────────
def test_merge_is_sparse(tmp_path):
    """오버레이에 없는 형제 키는 그대로다 — 섹션을 통째로 갈아 끼우지 않는다."""
    base = load_params_yaml()
    got = load_params_yaml(overlay=write(tmp_path, 'percep:\n  route_index_hops: 3\n'))
    assert got['percep']['route_index_hops'] == 3
    for k, v in base['percep'].items():
        if k != 'route_index_hops':
            assert got['percep'][k] == v


def test_empty_overlay_file_is_a_no_op(tmp_path):
    p = tmp_path / 'e.yaml'
    p.write_text('', encoding='utf-8')
    assert load_params_yaml(overlay=str(p)) == load_params_yaml()


# ── 배치 오버레이 자체의 건강 ───────────────────────────────────────────
def test_batch_overlay_only_names_existing_keys():
    """오버레이 파일이 실제 키만 가리키는가 (리팩터로 키가 사라지면 여기서 걸린다)."""
    load_params_yaml(overlay=str(BATCH_YAML))       # KeyError 면 실패


def test_batch_overlay_reproduces_the_batch_config():
    """켜면 배치가 쓰던 그 설정이 나온다."""
    cfg = load_params_yaml(overlay=str(BATCH_YAML))
    raw = yaml.safe_load(BATCH_YAML.read_text(encoding='utf-8'))
    for sec, keys in raw.items():
        for k, v in keys.items():
            assert cfg[sec][k] == v


def test_batch_overlay_stays_inside_the_controller_part():
    """경계 — 채점·시나리오 파트 키를 제어기 세션이 오버레이로 켜면 안 된다."""
    raw = yaml.safe_load(BATCH_YAML.read_text(encoding='utf-8'))
    forbidden = {'scoring', 'batch', 'gen_placement', 'gen_scenarios'}
    assert not (set(raw) & forbidden)
    for k in raw.get('route', {}):
        assert not k.startswith(('dp_', 'candidates_', 'finish_tail_', 'waypoint_'))
