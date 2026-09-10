"""
런 로그 첫 줄의 **설정 스냅샷**.

2026-09-10: `lane_map_shift_on_pick_enable` 이 꺼진 채 돈 것을 **폴백 경로를
역산해서야** 알았다 — `_lm_fb_hops` 는 `_owns_shift()` 안에서만 세팅되므로
2칸 폴백이 성립했다는 사실이 owns_shift 가 켜져 있었다는 증거이고, 그렇다면
`on_pick` 이 열렸어야 하는데 안 열렸으니 그 스위치가 off 였다. 대회날 이런
역산은 못 한다.

**기존 로그 소비자를 깨지 않는 것이 이 줄의 유일한 제약이다.**
score.py · batch_run.py · summarize_run.py 가 전부
`if '"raw"' not in line: continue` 로 거른다 → 이 줄에 `"raw"` 가 들어가면
그 세 도구가 설정 줄을 틱으로 착각한다.
"""
import json
import pathlib
import tempfile

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.logger import CONFIG_NUMS, Logger, _config_line

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = load_params_yaml(PARAMS_YAML)


def snap():
    return json.loads(_config_line(CFG))


def test_line_has_no_raw_key_so_consumers_skip_it():
    """이 파일이 지키는 핵심 — 세 도구의 필터를 그대로 통과(=무시)해야 한다."""
    line = _config_line(CFG)
    assert '"raw"' not in line
    assert '"event"' not in line          # summarize_run 은 이걸로 이벤트를 고른다
    assert '\n' not in line               # 한 줄


def test_consumer_filters_are_still_substring_based():
    """소비자 쪽 규약이 바뀌면 이 줄의 안전성 근거가 사라진다 — 같이 깨져야 한다."""
    for rel in ('tools/score.py', 'tools/batch_run.py', 'tools/summarize_run.py'):
        src = (ROOT / rel).read_text()
        assert '\'"raw"\' not in line' in src, rel


def test_every_enable_key_is_captured():
    """`*_enable` 은 이름을 몰라도 전부 잡힌다 (새 스위치가 자동으로 들어온다)."""
    d = snap()
    sw = d['switches']
    want = set()

    def walk(node, prefix):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, dict):
                walk(v, p)
            elif str(k).endswith('_enable'):
                want.add(p)
    walk(CFG, '')
    assert want, 'params 에 _enable 키가 하나도 없다?'
    assert want <= set(sw), sorted(want - set(sw))


def test_counts_are_consistent():
    d = snap()
    assert d['n_switch'] == len(d['switches'])
    assert d['n_on'] == len(d['on']) == sum(1 for v in d['switches'].values() if v)
    assert all(d['switches'][k] is True for k in d['on'])


def test_named_numbers_are_present_when_they_exist():
    d = snap()
    for key in CONFIG_NUMS:
        node = CFG
        for part in key.split('.'):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            assert key in d['nums'], key
            assert d['nums'][key] == node


def test_the_switches_this_session_added_are_visible():
    """오늘 만든 스위치가 스냅샷에 보여야 한다 — 그게 이 줄의 목적이다."""
    d = snap()
    for k in ('avoid_map.lane_map_shift_on_pick_enable',
              'avoid_map.lane_map_owns_shift_enable',
              'avoid_map.lane_map_fallback_side_enable',
              'speed.no_accel_toward_red_enable'):
        assert k in d['switches'], k
    assert 'percep.route_index_hops' in d['nums']
    assert 'speed.red_approach_min_kph' in d['nums']


def test_logger_writes_it_as_the_first_line():
    with tempfile.TemporaryDirectory() as td:
        p = str(pathlib.Path(td) / 'run.jsonl')
        lg = Logger(p, CFG)
        lg.close()
        first = pathlib.Path(p).read_text(encoding='utf-8').splitlines()[0]
        d = json.loads(first)
        assert d['kind'] == 'config'
        assert d['n_switch'] > 50


def test_snapshot_survives_a_full_log_queue():
    """큐는 가득 차면 버린다 — 그래서 이 줄은 큐를 열기 **전에** 동기로 쓴다."""
    src = (ROOT / 'vtd_adapter' / 'logger.py').read_text()
    i = src.index('_config_line(cfg)')
    j = src.index('self._q = queue.Queue(maxsize=2000)')
    assert i < j, '설정 줄이 큐 생성보다 먼저 쓰여야 한다'
