"""
batch_run 의 logs/batch/<ts>/report.txt 요약 표.

밤샘 배치를 아침에 한 화면에서 훑는 게 목적 — 표가 첫 화면에 오고, 실패·고감점이
위쪽에 오고, 주요위반이 잘리지 않아야 한다. 상세는 <시나리오>.score.txt /
report.json 에 그대로 남는다.

표 폭·표시 개수·제외 키는 params.yaml report.* 가 단일 출처다. 이 파일의 RCFG 는
"어떤 설정값이면 어떤 출력" 을 못 박는 테스트용이고, 실제 yaml 에 키가 다 있는지는
test_params_yaml_has_report_keys 가 따로 본다.
"""
import batch_run as br   # noqa: E402 (conftest 가 tools 경로 추가)
from vtd_adapter.config import load_params_yaml

RCFG = {'table_width': 100, 'name_w': 24, 'top_violations': 3,
        'violation_label_w': 12,
        'exclude_violations': ['not_finished', 'stall', 'off_route']}


def res(name, status='완주', deduction=0, violations=None, dist=100.0, t=60.0, kph=20.0,
        events=None):
    r = {'name': name, 'status': status, 'deduction': deduction,
         'violations': dict(violations or {}), 'dist_m': dist, 'time_s': t,
         'avg_kph': kph}
    if events is not None:                      # (성립, 전체[, 미도달])
        r['events_ok'], r['events_total'] = events[0], events[1]
        r['events_unreached'] = events[2] if len(events) > 2 else 0
    return r


def test_params_yaml_has_report_keys():
    """설정 단일 출처 — 코드에 기본값이 없으므로 yaml 에 키가 없으면 죽는다."""
    rc = load_params_yaml()['report']
    for k in RCFG:
        assert k in rc, k
    assert isinstance(rc['exclude_violations'], list)


# ── 주요위반 칸 ──────────────────────────────────────────────────────────
def test_top_violations_zero():
    assert br.top_violations({}, RCFG) == '-'
    assert br.top_violations(None, RCFG) == '-'


def test_top_violations_one():
    assert br.top_violations({'lane_departure': 2}, RCFG) == '차선이탈2'


def test_top_violations_five_keeps_three_and_counts_rest():
    """4개 이상이면 상위 3개 + ' 외N' — 동률은 검출 키 사전순."""
    v = {'lane_departure': 5, 'red_light': 4, 'speed': 3,
         'turn_signal': 2, 'center_line': 1}
    assert br.top_violations(v, RCFG) == '차선이탈5, 적신호통과4, 속도초과3 외2'


def test_top_violations_ties_break_by_key():
    v = {'stop_line_encroach': 2, 'red_light': 2, 'lane_departure': 2}
    assert br.top_violations(v, RCFG) == '차선이탈2, 적신호통과2, 정지선침범2'


def test_top_violations_excludes_status_duplicates():
    """미완주·정지 고착·경로 이탈은 상태 칸과 중복이라 빼고 센다."""
    v = {'not_finished': 1, 'stall': 1, 'off_route': 1, 'lane_departure': 2}
    assert br.top_violations(v, RCFG) == '차선이탈2'
    assert br.top_violations({'not_finished': 1, 'stall': 1}, RCFG) == '-'


def test_short_label_is_mechanical_shortening_of_score_label():
    """새 매핑 테이블 없이 score.LABEL 에서 괄호·공백만 걷어낸다."""
    assert br.short_label('speed', 10) == '속도초과'          # 속도 초과(법규)
    assert br.short_label('red_light', 10) == '적신호통과'    # 적신호 통과
    assert br.short_label('turn_signal', 20) == '차선변경지시등미점등'


def test_short_label_clips_with_ellipsis():
    # 폭은 표시칸 기준 — 한글 1자가 2칸이라 12칸이면 5자 + …
    assert br.short_label('turn_signal', 12) == '차선변경지…'
    assert br.short_label('turn_signal', 6) == '차선…'
    assert br.short_label('없는키', 10) == '없는키'           # LABEL 밖이면 키 그대로


# ── 시간·감점 칸 ─────────────────────────────────────────────────────────
def test_mmss():
    assert br.mmss(522.7) == '8:43'
    assert br.mmss(29.6) == '0:30'
    assert br.mmss(0) == '0:00'
    assert br.mmss(None) == '-'


def test_deduction_column_dash_when_scoring_missing():
    """채점이 안 돌았으면 감점 칸은 0 이 아니라 '-' — 무감점과 구분한다."""
    r = res('a', violations={'lane_departure': 1}); r['deduction'] = None
    assert br.render_report([r], RCFG).splitlines()[2].split()[5] == '-'
    assert br.render_report([res('a')], RCFG).splitlines()[2].split()[5] == '0'


# ── 정렬 + 요약 헤더 ─────────────────────────────────────────────────────
def test_sort_order_failures_then_worst_then_clean():
    rows = [res('깨끗', deduction=0),
            res('감점작음', deduction=-3),
            res('감점큼', deduction=-31),
            res('실패', status='stall', deduction=-6)]
    assert [r['name'] for r in sorted(rows, key=br.sort_key)] == \
        ['실패', '감점큼', '감점작음', '깨끗']


def test_sort_ties_break_by_name():
    rows = [res('b', deduction=-3), res('a', deduction=-3)]
    assert [r['name'] for r in sorted(rows, key=br.sort_key)] == ['a', 'b']


def test_summary_line():
    rows = [res(f'ok{i}') for i in range(2)]
    rows += [res('나쁨', deduction=-31), res('멈춤', status='stall', deduction=-6)]
    assert br.summary_line(rows, RCFG) == \
        '4건 · 완주 3 / stall 1 · 평균 감점 -9.2 · 최악 나쁨(-31)'


def test_summary_line_omits_worst_when_no_deduction():
    out = br.summary_line([res('ok')], RCFG)
    assert out == '1건 · 완주 1 · 평균 감점 0.0'


def test_summary_line_empty():
    assert br.summary_line([], RCFG) == '0건'


def test_summary_line_counts_collect_failures_last():
    """수집실패(deduction 없음)는 기존 필드 뒤에 붙고, 평균·최악에서는 빠진다."""
    bad = res('수집실패A'); bad['deduction'] = None
    bad2 = res('수집실패B', status='no-data'); bad2['deduction'] = None
    rows = [res('나쁨', deduction=-10), res('깨끗'), bad, bad2]
    assert br.summary_line(rows, RCFG) == (
        '4건 · 완주 3 / no-data 1 · 평균 감점 -5.0 · 최악 나쁨(-10) · 수집실패 2')


def test_summary_line_hides_collect_failures_when_zero():
    assert '수집실패' not in br.summary_line([res('ok', deduction=-3)], RCFG)


def test_summary_line_all_collect_failures():
    """전부 실패면 평균·최악 없이 수집실패만 — 0.0 으로 오해할 값을 안 만든다."""
    bad = res('a'); bad['deduction'] = None
    assert br.summary_line([bad], RCFG) == '1건 · 완주 1 · 수집실패 1'


# ── 표 전체 ──────────────────────────────────────────────────────────────
def test_report_columns_and_first_line_is_summary():
    out = br.render_report([res('시나리오A', deduction=-18, t=522.7, dist=3018.8,
                                kph=20.8, violations={'lane_departure': 2})], RCFG)
    lines = out.splitlines()
    assert lines[0].startswith('1건 · ')
    assert lines[1].split() == ['시나리오', '상태', '거리[m]', '시간',
                                '평균[km/h]', '감점', '이벤트', '주요위반']
    assert lines[2].split() == ['시나리오A', '완주', '3018.8', '8:43', '20.8',
                                '-18', '-', '차선이탈2']


def test_report_stays_within_table_width():
    """긴 이름 + 위반 다수여도 총폭 상한을 지킨다 (터미널에서 안 잘리도록)."""
    v = {k: 9 for k in ('lane_departure', 'red_light', 'speed', 'turn_signal',
                        'center_line', 'sidewalk', 'collision')}
    rows = [res('아주아주아주아주긴시나리오이름_00_직진좌회전우회전', deduction=-99,
                violations=v),
            res('짧음')]
    out = br.render_report(rows, RCFG)
    assert all(br.disp_w(ln) <= RCFG['table_width'] for ln in out.splitlines()[1:])


def test_long_scenario_name_is_ellipsized():
    long = '아주아주아주아주긴시나리오이름_00_직진좌회전우회전'
    out = br.render_report([res(long)], RCFG)
    cell = out.splitlines()[2].split()[0]
    assert cell.endswith('…') and br.disp_w(cell) <= RCFG['name_w']


def test_empty_results_renders_header_only():
    out = br.render_report([], RCFG)
    assert out.splitlines()[0] == '0건'
    assert out.splitlines()[1].split()[0] == '시나리오'


# ── 이벤트 조우 성립 칸 (tools/event_check.py 결과) ──────────────────────
def test_events_cell():
    assert br.events_cell(res('a', events=(4, 5))) == '4/5'
    assert br.events_cell(res('a', events=(0, 3))) == '0/3'
    assert br.events_cell(res('a', events=(0, 0))) == '-'      # 보행자 이벤트 없는 시나리오
    assert br.events_cell(res('a')) == '-'                     # 판정 안 돌았다
    r = res('a'); r['events_ok'] = r['events_total'] = None
    assert br.events_cell(r) == '?'                            # 판정 실패


def test_events_cell_separates_unreached_from_failure():
    """미완주 런에서 '0/4' 만 보이면 도달조차 못 한 이벤트가 실패로 읽힌다."""
    assert br.events_cell(res('a', events=(0, 4, 4))) == '0/4 (미도달4)'
    assert br.events_cell(res('a', events=(3, 5, 1))) == '3/5 (미도달1)'
    assert br.events_cell(res('a', events=(5, 5, 0))) == '5/5'      # 0 건이면 괄호 생략


def test_events_column_in_table():
    out = br.render_report([res('A', events=(4, 5))], RCFG)
    assert '4/5' in out.splitlines()[2]
    out = br.render_report([res('A', events=(0, 4, 4))], RCFG)
    assert '0/4 (미도달4)' in out.splitlines()[2]


def test_summary_line_aggregates_events():
    """밤샘 배치를 아침에 훑을 때 '몇 건 성립'이 한 줄에 보여야 한다."""
    rows = [res('a', events=(5, 6)), res('b', events=(3, 4)),
            res('c', events=(0, 0)), res('d')]
    assert '· 이벤트 8/10 성립' in br.summary_line(rows, RCFG)


def test_summary_line_aggregates_unreached():
    rows = [res('a', events=(5, 6, 1)), res('b', events=(0, 4, 4))]
    assert '· 이벤트 5/10 성립 (미도달5)' in br.summary_line(rows, RCFG)


def test_summary_line_omits_unreached_when_zero():
    rows = [res('a', events=(5, 6, 0)), res('b', events=(3, 4, 0))]
    s = br.summary_line(rows, RCFG)
    assert '· 이벤트 8/10 성립' in s and '미도달' not in s


def test_summary_line_omits_events_when_none():
    assert '이벤트' not in br.summary_line([res('a', events=(0, 0)), res('b')], RCFG)
