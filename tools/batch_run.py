"""
배치 회귀 실행기 — 시나리오 여러 개를 사람 손 없이 순차 실행하고 한 표로 정리.

    python3 tools/batch_run.py scenarios.json [--host 192.168.10.1] [--dry-run]
    python3 tools/batch_run.py scenarios/batch_보행자집중.json scenarios/batch_급정거집중.json
    python3 tools/batch_run.py 'scenarios/batch_*집중.json'      # glob 도 받는다 (합쳐 실행)

시나리오 목록(JSON, PyYAML 이 있으면 YAML 도):
    [{"name": "2_lead_brake",
      "vtd_xml_path": "/home/vtd/.../2_lead_brake.xml",   # VTD PC 기준 절대경로
      "route_csv": "waypoints/2_lead_brake.csv",          # 제어기 노트북 기준
      "timeout_s": 180}, ...]

한 사이클:
  1. build_route.py 로 logs/batch/<ts>/routes/route_{name}.pkl 생성
     (data/ 에는 안 남긴다 — 손으로 만든 pkl 만 두는 곳이다. 배치 산출 route 는
      로그와 같은 디렉터리에 묶여 사후 분석·replay 후 통째로 지운다)
  2. SCP(48179) 로 VTD 에 시나리오 로드 → Init → 팔로워 카메라(<Camera>,
     IG 표시 전용 — params camera.*) → Start   (실측 확인 2026-08-24)
  3. run_agent.py 를 서브프로세스로 (--route <routes>/route_{name}.pkl)
  4. 로그 꼬리를 감시해 종료 판정 (EndJudge — 완주 임계는 params.yaml 연동):
       완주        : route_s ≥ total − (stop_gap+앞범퍼+end_slack), 이후 v<0.5 또는 유예
       stall       : 정지인데 계획은 진행(v_target≥0.5)이 batch.stall_end_s 지속
       no_progress : route_s 무전진이 batch.no_progress_end_s 지속
       실패        : timeout_s 초과 / launch 프로세스 사망 / NO_DATA_S 로그 무변화
  5. launch SIGINT 종료, SCP Stop, 로그를 logs/batch/<ts>/{name}.jsonl 로 복사

실패해도 다음 시나리오로 진행하고 표에 사유를 남긴다 (조용히 넘기지 않는다).
Ctrl+C 는 launch 와 SCP 를 정리하고 부분 표를 출력한다.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time
import unicodedata

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'tools'))
sys.path.insert(0, str(_ROOT))

from scp_client import ScpClient                       # noqa: E402

STOP_GRACE_S = 10.0       # 완주 도달 후 정지(v<0.5)를 기다리는 최대 시간
NO_DATA_S = 20.0          # 로그가 이 시간 동안 안 자라면 no-data 실패
POLL_S = 2.0

# 완주 임계는 하드코딩하지 않는다 — vtd_adapter.config.end_margin_m(params.yaml) 과
# 공용 (summarize_run·score 도 같은 함수를 본다).
# (2026-08-25: stop_gap 1→4 튜닝 후 고정 임계 5 m 에 계획 정지점이 7.9 m 못 미쳐
#  정상 완주가 전부 timeout 처리된 사고)
from vtd_adapter.config import end_margin_m, load_params_yaml as load_cfg  # noqa: E402

DONE_RULE = ('완주 판정: 로그 ego.route_s ≥ route_{name}.pkl 총길이 − {margin:.1f} m '
             '(stop_gap+앞범퍼+여유, params.yaml 연동), 도달 후 v < 0.5 m/s 또는 '
             f'{STOP_GRACE_S:.0f} s 유예. 추가 종료: stall / no_progress (params batch.*)')


class EndJudge:
    """틱 스트림 → 종료 판정 (완주 / stall / no_progress). 순수 로직 — pytest 대상.

    · 완주       : route_s ≥ total − margin, 이후 v<0.5 즉시 또는 grace 유예
    · stall      : 정지(v<0.1)인데 계획은 진행(v_target≥0.5) — 전방 정지 사유가
                   없다는 뜻 — 이 상태가 stall_end_s 지속
    · no_progress: route_s 가 no_progress_end_s 동안 0.5 m 도 늘지 않음
                   (신호 대기 등 계획 정지 포함 — 임계는 적60 대기보다 길게 설정)
    """

    def __init__(self, total: float, margin: float, grace_s: float = STOP_GRACE_S,
                 stall_end_s: float = 20.0, no_progress_end_s: float = 120.0) -> None:
        self.total, self.margin, self.grace = total, margin, grace_s
        self.stall_s, self.no_prog_s = stall_end_s, no_progress_end_s
        self.reached_at = None
        self.best_rs = -math.inf
        self.progress_t = None
        self.stall_t = None

    def feed(self, now: float, route_s: float, v: float, v_target: float) -> str | None:
        if self.progress_t is None:
            self.progress_t = now
        # 완주
        if route_s >= self.total - self.margin:
            if self.reached_at is None:        # `or now` 는 now=0.0 을 falsy 로 오판
                self.reached_at = now
            if v < 0.5 or now - self.reached_at > self.grace:
                return '완주'
        # 전진 감시
        if route_s > self.best_rs + 0.5:
            self.best_rs = route_s
            self.progress_t = now
        elif now - self.progress_t > self.no_prog_s:
            return 'no_progress'
        # 계획은 가려는데 차가 못 가는 상태
        if v < 0.1 and v_target >= 0.5:
            self.stall_t = self.stall_t if self.stall_t is not None else now
            if now - self.stall_t > self.stall_s:
                return 'stall'
        else:
            self.stall_t = None
        return None


def _load_one(path: str) -> list[dict]:
    text = open(path, encoding='utf-8').read()
    if path.endswith(('.yml', '.yaml')):
        import yaml                                    # 없으면 그대로 ImportError
        items = yaml.safe_load(text)
    else:
        items = json.loads(text)
    out = []
    for i, it in enumerate(items):
        for k in ('name', 'vtd_xml_path', 'route_csv'):
            if k not in it:
                raise SystemExit(f'{path} 시나리오 [{i}] 에 {k} 가 없다: {it}')
        it.setdefault('timeout_s', 180.0)
        out.append(it)
    return out


def load_scenarios(paths: list[str]) -> list[dict]:
    """목록 파일 여러 개(glob 포함)를 합쳐 하나의 실행 목록으로.

    이름 중복 검사는 **통합 후** 기준이다 — 같은 시나리오가 두 목록에 들어 있으면
    (batch_all.json 과 주제별 json 을 같이 주는 실수 등) 여기서 잡힌다.
    """
    import glob as _glob
    files: list[str] = []
    for p in paths:
        hits = sorted(_glob.glob(p))
        if not hits:
            raise SystemExit(f'시나리오 목록이 없다: {p}')
        files += hits
    seen_files = set()
    out: list[dict] = []
    for f in files:
        if f in seen_files:                            # 같은 파일이 두 패턴에 걸린 경우
            continue
        seen_files.add(f)
        out += _load_one(f)
    names = [it['name'] for it in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f'통합 목록에서 시나리오 이름이 중복된다: {dup}\n'
                         f'  (입력 파일: {", ".join(files)})')
    return out


def tail_tick(path: pathlib.Path) -> dict | None:
    """jsonl 꼬리에서 마지막 **완전한** 틱 레코드 하나."""
    try:
        size = path.stat().st_size
        with open(path, 'rb') as f:
            f.seek(max(0, size - 65536))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if '"raw"' not in line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue                                   # 잘린 마지막 줄
    return None


# ── IG 팔로워 뷰 카메라 (표시 전용 — GT·제어·채점 무관) ───────────────────
def build_camera_scp(cfg: dict) -> str | None:
    """IG 팔로워 뷰 <Camera> SCP 문자열. camera.enabled=false 면 None.

    GUI 프리셋 "Relative - follower view" 와 동일 뷰 (VTD 2025.2 Doc/SCP_HTML
    실기 확인 2026-08-27): PosRelative 는 자차 로컬 [m], ViewRelative 각도는
    [rad], Show Owner 는 <Camera showOwner="true|false"> 속성, <Set/> 이 적용.
    값은 params.yaml camera.* 가 단일 출처 — 키가 없으면 KeyError 로 죽는 게 맞다.
    """
    cam = cfg['camera']
    if not cam['enabled']:
        return None
    return (f'<Camera name="followCam" showOwner="{str(bool(cam["show_owner"])).lower()}">'
            f'<PosRelative player="{cam["player"]}" dx="{float(cam["dx"]):.2f}" '
            f'dy="{float(cam["dy"]):.2f}" dz="{float(cam["dz"]):.2f}"/>'
            f'<ViewRelative dh="{float(cam["dh"]):.4f}" dp="{float(cam["dp"]):.4f}" '
            f'dr="{float(cam["dr"]):.4f}"/>'
            f'<Set/></Camera>')


# ── 표 렌더 (콘솔과 report.txt 가 같은 함수를 쓴다 — 중복 구현 금지) ──────
OVERFLOW_W = 24     # 무한정 길어질 수 있는 열(shield·위반내역)의 폭 상한 [표시칸]


def disp_w(s) -> int:
    """표시 폭 — 한글·전각(east_asian_width W/F)은 2칸.
    str.ljust 는 한글을 1칸으로 세므로 그대로 쓰면 표가 어긋난다."""
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(s))


def _pad(s, w: int) -> str:
    return str(s) + ' ' * max(0, w - disp_w(s))


def render_table(hdr: list, rows: list, overflow: tuple = ()) -> str:
    """폭 정렬 텍스트 표. 컬럼 폭은 실제 내용의 최대 표시폭에 동적으로 맞춘다.

    overflow 에 든 열 이름은 폭을 OVERFLOW_W 로 상한한다 — 그보다 긴 셀은
    자르지 않고 **그 행만** 넘치게 둔다 (앞 열들의 정렬은 유지된다).
    """
    widths = []
    for i, h in enumerate(hdr):
        w = max(disp_w(h), *(disp_w(r[i]) for r in rows)) if rows else disp_w(h)
        if h in overflow:
            w = min(w, max(disp_w(h), OVERFLOW_W))
        widths.append(w)
    lines = ['  '.join(_pad(c, widths[i]) for i, c in enumerate(hdr)).rstrip()]
    lines += ['  '.join(_pad(c, widths[i]) for i, c in enumerate(row)).rstrip()
              for row in rows]
    return '\n'.join(lines)


class Runner:
    def __init__(self, args) -> None:
        self.args = args
        self.launch: subprocess.Popen | None = None
        self.scp: ScpClient | None = None
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = _ROOT / 'logs' / 'batch' / ts
        self.results: list[dict] = []
        # 종료 판정 값은 컨트롤러와 같은 params.yaml 에서 (하드코딩 금지)
        self.cfg = cfg = load_cfg()
        self.margin = end_margin_m(cfg)
        b = cfg.get('batch', {})
        self.stall_end_s = float(b.get('stall_end_s', 20.0))
        self.no_progress_end_s = float(b.get('no_progress_end_s', 120.0))

    # ── 정리 (Ctrl+C 포함 모든 경로에서 호출) ─────────────────────────────
    def cleanup(self) -> None:
        if self.launch is not None and self.launch.poll() is None:
            try:
                # SIGINT(정상 종료) → SIGTERM → SIGKILL 순으로 격상.
                # launch 가 SIGINT/SIGTERM 에 무반응인 경우가 있어 최종 KILL 보장.
                os.killpg(self.launch.pid, signal.SIGINT)
                try:
                    self.launch.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.launch.pid, signal.SIGTERM)
                    try:
                        self.launch.wait(timeout=4.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.launch.pid, signal.SIGKILL)
                        self.launch.wait(timeout=5.0)
            except (ProcessLookupError, PermissionError):
                pass
        self.launch = None
        if self.scp is not None:
            try:
                self.scp.stop()
                self.scp.close()
            except OSError:
                pass
            self.scp = None

    # ── IG 카메라 ─────────────────────────────────────────────────────────
    def _apply_camera(self):
        """팔로워 뷰 SCP 1회 전송. 표시 전용 — 실패해도 배치는 계속 (경고만).

        반환값이 run 메타(res['camera_sent'])에 남는다:
        True 전송 / False 실패(무시) / None 비활성(camera.enabled=false).
        """
        xml = build_camera_scp(self.cfg)
        if xml is None:
            return None
        try:
            self.scp.send(xml)
            return True
        except (OSError, RuntimeError) as e:
            print(f'  ⚠ 카메라 뷰 SCP 실패 — 무시하고 진행: {e}')
            return False

    # ── VTD 9910 예열 확인 ────────────────────────────────────────────────
    def _wait_9910(self, warmup_s: float) -> float | None:
        """9910 에서 1 프레임이라도 받을 때까지 대기 → 걸린 시간 [s], 초과 시 None.

        확인 후 즉시 닫는다 (vtd_bridge 가 붙을 자리를 오래 점유하지 않는다).
        """
        import socket
        t0 = time.monotonic()
        while time.monotonic() - t0 < warmup_s:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((self.args.host, 9910))
                data = s.recv(4096)
                s.close()
                if data:
                    time.sleep(0.5)      # 프로브가 소켓을 놓은 뒤 컨트롤러가 붙게
                    return time.monotonic() - t0
            except OSError:
                try:
                    s.close()
                except OSError:
                    pass
            time.sleep(2.0)
        return None

    # ── 한 시나리오 ───────────────────────────────────────────────────────
    def run_one(self, sc: dict) -> dict:
        name = sc['name']
        res = {'name': name, 'status': '?', 'log': None}
        # route pkl 은 이 배치의 로그 디렉터리에 — data/ 를 오염시키지 않는다
        route_pkl = self.out_dir / 'routes' / f'route_{name}.pkl'

        # 1) route 빌드 — 공용 data/route.pkl 은 건드리지 않는다
        build_cmd = [sys.executable, str(_ROOT / 'tools' / 'build_route.py'),
                     str(_ROOT / 'data' / 'lane_graph.pkl'), sc['route_csv'],
                     '-o', str(route_pkl)]
        agent_log = self.out_dir / f'{name}.jsonl'
        launch_cmd = [sys.executable, str(_ROOT / 'run_agent.py'),
                      '--route', str(route_pkl),
                      '--graph', str(_ROOT / 'data' / 'lane_graph.pkl'),
                      '--host', self.args.host,
                      '--log', str(agent_log)]
        if self.args.dry_run:
            print(f'\n[dry-run] {name}')
            print('  route :', ' '.join(build_cmd))
            print('  scp   : Load(%r) → Init → Camera → Start @ %s:48179'
                  % (sc['vtd_xml_path'], self.args.host))
            print('  camera:', build_camera_scp(self.cfg) or '비활성 (camera.enabled=false)')
            print('  launch:', ' '.join(launch_cmd))
            print(f'  종료   : {DONE_RULE.format(name=name, margin=self.margin)} '
                  f'/ timeout {sc["timeout_s"]}s')
            res['status'] = 'dry-run'
            return res

        route_pkl.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(build_cmd, capture_output=True, text=True, timeout=120)
        (self.out_dir / f'{name}.build_route.txt').write_text(cp.stdout + cp.stderr)
        if cp.returncode != 0 or not route_pkl.exists():
            res['status'] = 'route 실패'
            # ⚠ 요약 한 줄을 표 비고에 노출 — 파일을 열지 않아도 원인이 보이게
            warn = next((line.strip() for line in (cp.stdout + cp.stderr).splitlines()
                         if '⚠' in line), '')
            if warn:
                res['collect_error'] = warn[:140]
            print(f'  ✗ build_route 실패 (rc={cp.returncode}) — {warn or f"{name}.build_route.txt 참고"}')
            return res
        import pickle
        total = float(pickle.load(open(route_pkl, 'rb'))['total_length'])
        res['route_total_m'] = round(total, 1)

        # 2) VTD 로드→Init→(카메라)→Start — scp_client.load_and_run 을 단계로
        #    분해했다: IG 팔로워 뷰는 시나리오 Load 마다 초기화되므로 Init 완료
        #    후·Start 직전에 시나리오당 1회 다시 적용해야 한다 (표시 전용).
        try:
            self.scp = ScpClient(self.args.host, verbose=False).connect()
            settle = self.args.settle_s
            self.scp.load_scenario(sc['vtd_xml_path'])
            self.scp.poll(settle)
            self.scp.init()
            self.scp.poll(settle)                  # Init 완료 대기 (기존 settle 관례)
            res['camera_sent'] = self._apply_camera()
            self.scp.start()
            self.scp.poll(settle)
        except OSError as e:
            res['status'] = f'SCP 실패: {e}'
            self.cleanup()
            return res

        # 2.5) VTD 예열 대기 — 특히 기동 직후 첫 시나리오는 로드/Init 에 수십 초가
        # 걸린다. 9910 이 실제로 프레임을 흘릴 때까지 기다린 뒤에 컨트롤러를 띄우고
        # timeout 시계를 돌린다 (안 기다리면 no-data 로 오판).
        waited = self._wait_9910(float(self.args.vtd_warmup_s))
        if waited is None:
            res['status'] = f'VTD 9910 무응답 ({self.args.vtd_warmup_s:.0f}s 예열 대기 초과)'
            self.cleanup()
            return res
        if waited > 1.0:
            print(f'    VTD 예열 {waited:.0f}s 후 9910 수신 시작')

        # 3) 컨트롤러 실행 (새 세션 = 프로세스 그룹 → SIGINT 로 통째로 정리)
        launch_out = open(self.out_dir / f'{name}.launch.txt', 'w')
        self.launch = subprocess.Popen(launch_cmd, cwd=str(_ROOT), start_new_session=True,
                                       stdout=launch_out, stderr=subprocess.STDOUT)

        # 4) 로그 감시 — --log 로 지정한 파일이 이 시나리오의 로그다
        log_path = None
        t0 = time.monotonic()
        deadline = t0 + float(sc['timeout_s'])
        last_size, last_grow = -1, time.monotonic()
        judge = EndJudge(total, self.margin, STOP_GRACE_S,
                         self.stall_end_s, self.no_progress_end_s)
        status = 'timeout'
        while time.monotonic() < deadline:
            time.sleep(POLL_S)
            if self.launch.poll() is not None:
                status = f'launch 사망 (rc={self.launch.returncode})'
                break
            if log_path is None:
                if agent_log.exists():
                    log_path = agent_log
                    print(f'    로그: {log_path.name}')
                elif time.monotonic() - t0 > NO_DATA_S:
                    status = 'no-data (로그 파일이 생기지 않음)'
                    break
                continue
            size = log_path.stat().st_size
            if size != last_size:
                last_size, last_grow = size, time.monotonic()
            elif time.monotonic() - last_grow > NO_DATA_S:
                status = f'no-data ({NO_DATA_S:.0f}s 로그 정체)'
                break
            tick = tail_tick(log_path)
            if tick is None:
                continue
            verdict = judge.feed(time.monotonic(),
                                 float(tick['ego']['route_s']),
                                 float(tick['ego']['speed']),
                                 float(tick['decision'].get('v_target') or 0.0))
            if verdict is not None:
                status = verdict
                break

        # 5) 정리 + 로그 회수 (--log 로 이미 out_dir 에 쓰였다)
        self.cleanup()
        launch_out.close()
        res['status'] = status
        if agent_log.exists():
            res['log'] = str(agent_log)
        return res

    # ── 결과 수집 ─────────────────────────────────────────────────────────
    def collect(self, res: dict) -> None:
        if not res.get('log'):
            return
        import pickle
        lg = route = None
        try:
            from summarize_run import load_map
            lg, _ = load_map()
            route = pickle.load(open(self.out_dir / 'routes' / f"route_{res['name']}.pkl", 'rb'))
        except Exception as e:                          # noqa: BLE001
            res['collect_error'] = f'map: {type(e).__name__}: {e}'
        try:
            from summarize_run import summarize
            s = summarize(res['log'], lg, route)
            f = s.get('finish', {})
            res.update(done=f.get('done'), dist_m=f.get('dist_m'), time_s=f.get('time_s'),
                       avg_kph=f.get('avg_kph'), end_stopped=f.get('end_stopped'),
                       overspeed_ticks=s.get('overspeed', {}).get('ticks'),
                       resets=len(s.get('resets', [])))
            ticks = [json.loads(l) for l in open(res['log']) if '"raw"' in l]
            res['estop_ticks'] = sum(1 for t in ticks if t['decision']['state'] == 'E_STOP')
            fired: dict = {}
            for t in ticks:
                for k in (t['decision']['reasons'].get('shield') or {}):
                    fired[k] = fired.get(k, 0) + 1
            res['shield'] = fired
            dmin = math.inf
            for t in ticks:
                ex, ey = t['ego']['x'], t['ego']['y']
                for o in t['objects']:
                    dmin = min(dmin, math.hypot(o['x'] - ex, o['y'] - ey))
            res['min_obj_dist_m'] = None if math.isinf(dmin) else round(dmin, 2)
        except Exception as e:                          # noqa: BLE001 — 수집 실패도 표에 남긴다
            res['collect_error'] = res.get('collect_error', '') + f' {type(e).__name__}: {e}'
        # 위반 검출 (tools/score.py — 배점 없음, 건수만). 전문은 {name}.score.txt 로.
        try:
            import score as score_tool
            rep = score_tool.analyze(res['log'], load_cfg(), lg, route)
            (self.out_dir / f"{res['name']}.score.txt").write_text(
                score_tool.render(rep), encoding='utf-8')
            res['n_violations'] = rep['n_violations']
            res['violations'] = {k: d['count'] for k, d in rep['violations'].items()
                                 if d['count'] and k not in score_tool.INFO_KEYS}
            # 정지선 정지 지표 — report 에 실어 **로그가 지워져도 남긴다**
            ticks = score_tool.load_ticks(res['log'])
            res['stops'] = score_tool.stop_metrics(ticks, ticks[0]['t']) if ticks else []
        except Exception as e:                          # noqa: BLE001
            res['n_violations'] = None
            res['collect_error'] = res.get('collect_error', '') + f' score:{e}'

    # ── 정지 지표 (영구 기록) ──────────────────────────────────────────────
    STOP_RULE = ('정지선 정지 지표: slf = 앞범퍼→정지선 [m] (음수 = 선 앞). '
                 '계획값 = −speed.stop_gap_stopline_m. '
                 '전환 = 감속 커맨드 방향 전환 수 (단조 감속이면 작다).\n'
                 '녹색후출발 = 녹색 전환 → v>0.3 m/s 까지 [s]. **관측 항목이지 '
                 '판정 항목이 아니다** — _stopline_hold 리필 수정(보류, '
                 'docs/BACKLOG.md) 재개 여부를 이 수치로 판단한다. 참고 기준 <0.3 s.')

    def stop_report(self) -> str:
        """정지 이벤트별 지표 표. 로그를 지워도 report.txt 에 남는다."""
        rows = []
        for r in self.results:
            for st in (r.get('stops') or []):
                rows.append([r['name'], st['t'], st['route_s'], st['slf_m'],
                             st['approach_kph'], st['cmd_reversals'], st['hold_s'],
                             '-' if st.get('green_delay_s') is None else st['green_delay_s'],
                             ','.join(map(str, st['ctrl_ids'])),
                             ','.join(map(str, st['states'])) or '-'])
        if not rows:
            return ''
        slf = [x[3] for x in rows]
        gd = [x[7] for x in rows if x[7] != '-']
        summ = ('  판정: 침범(slf>0) %d건  slf 평균 %+.2f  산포(p-p) %.2f  최대 %+.2f\n'
                '  관측: 전환 중앙값 %d  녹색후출발 %s'
                % (sum(1 for v in slf if v > 0), sum(slf) / len(slf),
                   max(slf) - min(slf), max(slf),
                   sorted(x[5] for x in rows)[len(rows) // 2],
                   ('평균 %.2f s (n=%d)' % (sum(gd) / len(gd), len(gd))) if gd else '—'))
        hdr = ['시나리오', 't[s]', 'route_s', 'slf[m]', '접근[km/h]', '전환',
               '유지[s]', '녹색후출발[s]', 'ctrl', '신호']
        return self.STOP_RULE + '\n' + render_table(hdr, rows) + '\n' + summ

    # ── 표 ────────────────────────────────────────────────────────────────
    def report(self) -> str:
        hdr = ['시나리오', '상태', '완주', '거리[m]', '시간[s]', '평균[km/h]',
               '최소객체거리[m]', 'E_STOP', 'shield', '위반', '위반내역', '비고']
        rows = []
        for r in self.results:
            col = 'Y' if r.get('min_obj_dist_m') is not None and r['min_obj_dist_m'] < 2.5 else ''
            nv = r.get('n_violations')
            detail = ','.join(f'{k}:{v}' for k, v in (r.get('violations') or {}).items())
            rows.append([
                r['name'], r['status'],
                {True: 'O', False: 'X'}.get(r.get('done'), '-'),
                r.get('dist_m', '-'), r.get('time_s', '-'), r.get('avg_kph', '-'),
                ('%s%s' % (r.get('min_obj_dist_m', '-'), ' ⚠충돌?' if col else '')),
                r.get('estop_ticks', '-'),
                ','.join(f'{k}:{v}' for k, v in (r.get('shield') or {}).items()) or '-',
                '-' if nv is None else nv,
                # 검출이 돌았고 위반이 없으면 '0건' — '-' 는 "검출 못 함"에만 쓴다
                detail or ('0건' if nv == 0 else '-'),
                r.get('collect_error', ''),
            ])
        return render_table(hdr, rows, overflow=('shield', '위반내역'))

    # ── 전체 ──────────────────────────────────────────────────────────────
    # ── 실행 전 검사: VTD PC 에 시나리오 xml 이 실제로 있는가 ────────────
    def precheck_vtd_paths(self, scenarios: list[dict]) -> None:
        """첫 시나리오의 vtd_xml_path 를 ssh ls 로 실존 확인.

        과거 사고: /home/vtd/... 프리픽스로 생성된 목록이 실기에 존재하지 않아
        5개 중 2개가 로드 실패(no-data)로 시간만 태웠다. 없으면 즉시 중단한다.
        """
        path = scenarios[0]['vtd_xml_path']
        user = path.split('/')[2] if path.startswith('/home/') else 'mjw'
        target = self.args.ssh or f'{user}@{self.args.host}'
        cp = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             target, 'ls', path],
                            capture_output=True, text=True, timeout=15)
        if cp.returncode != 0:
            raise SystemExit(
                f'[중단] VTD PC({target})에 {path} 가 없다 — 복사 안 됨 또는 경로 불일치.\n'
                f'       scenarios/ 를 VTD PC 로 복사했는지, gen_scenarios --vtd-dir 프리픽스가\n'
                f'       실제 복사 위치와 같은지 확인할 것. (ssh 출력: {cp.stderr.strip() or cp.stdout.strip()})')
        print(f'VTD 경로 확인: {target}:{path} OK')

    def run(self) -> int:
        scenarios = load_scenarios(self.args.scenarios)
        if not self.args.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.precheck_vtd_paths(scenarios)
        print(f'배치 {len(scenarios)}건, host={self.args.host}')
        print(DONE_RULE.format(name='<name>', margin=self.margin))
        try:
            for i, sc in enumerate(scenarios, 1):
                print(f'\n[{i}/{len(scenarios)}] {sc["name"]}  (timeout {sc["timeout_s"]}s)')
                try:
                    res = self.run_one(sc)
                except KeyboardInterrupt:
                    raise
                except Exception as e:                  # noqa: BLE001 — 실패해도 다음으로
                    res = {'name': sc['name'], 'status': f'예외: {type(e).__name__}: {e}', 'log': None}
                    self.cleanup()
                self.results.append(res)
                if not self.args.dry_run:
                    self.collect(res)
                print(f'  → {res["status"]}')
                time.sleep(self.args.pause_s)
        except KeyboardInterrupt:
            print('\n[중단] 정리 중…')
            self.cleanup()
        table = self.report()
        print('\n' + '=' * 100)
        print(DONE_RULE.format(name='<name>', margin=self.margin))
        print(table)
        stops = self.stop_report()
        if stops:
            print('\n' + stops)
        if not self.args.dry_run:
            (self.out_dir / 'report.txt').write_text(
                DONE_RULE.format(name='<name>', margin=self.margin) + '\n' + table + '\n'
                + ('\n' + stops + '\n' if stops else ''), encoding='utf-8')
            (self.out_dir / 'report.json').write_text(
                json.dumps(self.results, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'\n리포트: {self.out_dir}/report.txt (.json, 개별 로그·score 전문 포함)')
        return 0 if all(r['status'] in ('완주', 'dry-run') for r in self.results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='배치 회귀 실행기 (SCP 원격 제어)')
    ap.add_argument('scenarios', nargs='+',
                    help='시나리오 목록 json/yaml — 여러 개·glob 가능, 합쳐서 순차 실행')
    ap.add_argument('--host', default='192.168.10.1', help='VTD 주소 (SCP 48179 + 9910)')
    ap.add_argument('--ssh', default=None,
                    help='실행 전 시나리오 실존 확인용 ssh 대상 (기본: <vtd_xml_path 의 홈 사용자>@host)')
    ap.add_argument('--settle-s', type=float, default=3.0, help='SCP 명령 간 대기')
    ap.add_argument('--vtd-warmup-s', type=float, default=120.0,
                    help='시나리오 Start 후 9910 첫 프레임까지 최대 대기 (VTD 기동 직후 대비)')
    ap.add_argument('--pause-s', type=float, default=5.0,
                    help='시나리오 간 대기 (9910 소켓 TIME_WAIT 정리 여유)')
    ap.add_argument('--dry-run', action='store_true', help='실행 없이 계획만 출력')
    a = ap.parse_args(argv)
    return Runner(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
