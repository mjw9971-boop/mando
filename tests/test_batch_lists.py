"""
gen_scenarios batch 목록 재생성 (2026-08-27 덮어쓰기 사고 회귀).

사고: 목록을 "이번 호출분" 메모리로 쓰던 탓에 주제를 차례로 생성하면
batch_all.json 에 마지막 주제만 남았다. 디스크의 <주제>/*.yaml 이 단일 출처다.
"""
import json
import os
import pathlib

import pytest
import yaml

import gen_scenarios as gs   # noqa: E402 (conftest 가 tools 경로 추가)

VTD_DIR = '/home/mjw/scenarios'


def make_theme(out_dir: pathlib.Path, theme: str, names: list, t0: float):
    d = out_dir / theme
    d.mkdir(parents=True)
    os.utime(d)                        # 생성 시각 근사 — ctime 은 아래에서 강제 못 하므로
    for i, name in enumerate(names):
        (d / f'{name}.yaml').write_text(
            yaml.safe_dump({'name': name, 'theme': theme, 'timeout_s': 180 + i},
                           allow_unicode=True), encoding='utf-8')
    return d


def test_rebuild_merges_all_themes(tmp_path):
    """주제 2개(하나는 두 번 생성돼 번호가 이어진 꼴) → all = 합, 순서·중복 검증."""
    # 주제A 를 먼저 생성 (두 번의 생성이 번호로 이어진 상황: 01~02 + 03~04)
    make_theme(tmp_path, '주제A', ['주제A_01_기본', '주제A_02_직진',
                                   '주제A_03_기본', '주제A_04_직진'], 0)
    make_theme(tmp_path, '주제B', ['주제B_01_기본'], 1)

    n_all, n_themes = gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert (n_all, n_themes) == (5, 2)

    all_items = json.loads((tmp_path / 'batch_all.json').read_text())
    names = [it['name'] for it in all_items]
    assert names == ['주제A_01_기본', '주제A_02_직진', '주제A_03_기본',
                     '주제A_04_직진', '주제B_01_기본']        # 주제=생성순, 안=번호순
    assert len(set(names)) == len(names)                      # 중복 없음

    a_items = json.loads((tmp_path / 'batch_주제A.json').read_text())
    assert len(a_items) == 4                                  # 이전 생성분 포함 전체
    # batch_run 스키마 그대로 (필수 키 + 경로 규칙)
    it = a_items[0]
    assert it == {'name': '주제A_01_기본',
                  'vtd_xml_path': f'{VTD_DIR}/주제A/주제A_01_기본.xml',
                  'route_csv': f'{tmp_path.name}/주제A/주제A_01_기본.csv',
                  'timeout_s': 180}


def test_rebuild_is_idempotent_and_reads_disk(tmp_path):
    """재호출해도 결과 동일 — 디스크가 출처이므로 호출 이력과 무관하다."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    make_theme(tmp_path, '주제B', ['주제B_01_기본'], 1)       # 나중 주제 추가 생성
    n_all, _ = gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert n_all == 2                                         # A 가 사라지지 않는다
    names = [it['name'] for it in json.loads((tmp_path / 'batch_all.json').read_text())]
    assert names == ['주제A_01_기본', '주제B_01_기본']


def test_duplicate_names_write_nothing(tmp_path):
    make_theme(tmp_path, '주제A', ['같은이름'], 0)
    make_theme(tmp_path, '주제B', ['같은이름'], 1)
    with pytest.raises(SystemExit):
        gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert not (tmp_path / 'batch_all.json').exists()
    assert not (tmp_path / 'batch_주제A.json').exists()       # 부분 쓰기도 없다


def test_unknown_theme_leaves_lists_untouched(tmp_path):
    """모르는 주제로 죽는 경로는 목록을 건드리지 않는다."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    before = (tmp_path / 'batch_all.json').read_text()
    with pytest.raises(SystemExit):
        gs.main(['이런주제없다', '--out-dir', str(tmp_path)])
    assert (tmp_path / 'batch_all.json').read_text() == before


def test_rebuild_lists_cli(tmp_path):
    """--rebuild-lists: 생성 없이 목록만 재생성 (lane_graph 불필요)."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    assert gs.main(['--rebuild-lists', '--out-dir', str(tmp_path)]) == 0
    assert len(json.loads((tmp_path / 'batch_all.json').read_text())) == 1
