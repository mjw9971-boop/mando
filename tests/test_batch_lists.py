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


# ── quick_list.txt → batch_quick.json ────────────────────────────────────────
# 낮에 21건(8.2 h)을 못 돌리므로 목적별 7개만 고른 짧은 목록을 쓴다. 손으로 유지한
# 목록은 시나리오 재생성 때마다 낡으므로(scenarios/ 는 gitignore) 이름만 추적하고
# 나머지는 매번 batch_all 에서 다시 가져온다.

def write_quick(out_dir: pathlib.Path, text: str):
    (out_dir / 'quick_list.txt').write_text(text, encoding='utf-8')


def test_quick_list_selects_named_items_in_listed_order(tmp_path):
    """quick_list 순서를 그대로 쓴다 — 낮 배치 실행 순서를 사람이 정하는 파일이다."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본', '주제A_02_직진',
                                   '주제A_03_좌회전'], 0)
    write_quick(tmp_path, '# 주석\n\n주제A_03_좌회전  # 뒤 주석\n주제A_01_기본\n')

    gs.rebuild_batch_lists(tmp_path, VTD_DIR)

    items = json.loads((tmp_path / 'batch_quick.json').read_text())
    assert [it['name'] for it in items] == ['주제A_03_좌회전', '주제A_01_기본']
    assert items[0]['timeout_s'] == 182                  # batch_all 항목을 그대로 재사용
    assert items[0]['vtd_xml_path'] == f'{VTD_DIR}/주제A/주제A_03_좌회전.xml'


def test_quick_list_absent_writes_nothing(tmp_path):
    """목록 파일이 없으면 batch_quick.json 을 만들지 않는다 (빈 목록도 아니다)."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert not (tmp_path / 'batch_quick.json').exists()


def test_quick_list_missing_name_warns_and_keeps_rest(tmp_path, capsys):
    """이름이 안 맞아도 실패시키지 않는다 — 목록 하나가 시나리오 생성 전체를 막으면 안 된다."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    write_quick(tmp_path, '주제A_01_기본\n사라진_시나리오\n')

    n_all, _ = gs.rebuild_batch_lists(tmp_path, VTD_DIR)

    assert n_all == 1                                    # 전체 목록은 정상 생성
    items = json.loads((tmp_path / 'batch_quick.json').read_text())
    assert [it['name'] for it in items] == ['주제A_01_기본']
    assert '사라진_시나리오' in capsys.readouterr().out   # 조용히 빠지지 않는다


def test_quick_list_is_regenerated_after_scenarios_change(tmp_path):
    """재생성 때마다 최신 batch_all 을 다시 읽는다 — timeout 이 바뀌면 따라간다."""
    make_theme(tmp_path, '주제A', ['주제A_01_기본'], 0)
    write_quick(tmp_path, '주제A_01_기본\n')
    gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert json.loads((tmp_path / 'batch_quick.json').read_text())[0]['timeout_s'] == 180

    (tmp_path / '주제A' / '주제A_01_기본.yaml').write_text(
        yaml.safe_dump({'name': '주제A_01_기본', 'theme': '주제A', 'timeout_s': 999},
                       allow_unicode=True), encoding='utf-8')
    gs.rebuild_batch_lists(tmp_path, VTD_DIR)
    assert json.loads((tmp_path / 'batch_quick.json').read_text())[0]['timeout_s'] == 999


def test_repo_quick_list_names_all_exist():
    """저장소의 quick_list.txt 7건이 실제 batch_all 에 있는지 — 이름 오타 방지."""
    scen = pathlib.Path(gs.ROOT) / 'scenarios'
    names = gs.read_quick_names(scen)
    if names is None or not (scen / 'batch_all.json').exists():
        pytest.skip('scenarios/ 미생성 환경')
    have = {it['name'] for it in json.loads((scen / 'batch_all.json').read_text())}
    assert len(names) == 7
    assert [n for n in names if n not in have] == []
