"""
배치 회귀 실행기 — 시나리오 여러 개를 사람 손 없이 순차 실행하고 한 표로 정리.

    python3 tools/batch_run.py scenarios.json [--host 192.168.10.1] [--dry-run]

시나리오 목록(JSON, PyYAML 이 있으면 YAML 도):
    [{"name": "2_lead_brake",
      "vtd_xml_path": "/home/vtd/.../2_lead_brake.xml",   # VTD PC 기준 절대경로
      "route_csv": "waypoints/2_lead_brake.csv",          # 제어기 노트북 기준
      "timeout_s": 180}, ...]

한 사이클:
  1. build_route.py 로 data/route_{name}.pkl 생성 (**공용 route.pkl 은 건드리지 않는다**)
  2. SCP(48179) 로 VTD 에 시나리오 로드 → Init → Start   (실측 확인 2026-08-24)
  3. ros2 launch 를 서브프로세스로 (route:=data/route_{name}.pkl)
  4. 로그 꼬리를 감시해 종료 판정:
       완주   : route_s ≥ total_length − END_MARGIN_M, 이후 v < 0.5 m/s 또는
                STOP_GRACE_S 유예   ← 완주 판정 방식은 리포트 헤더에도 기록된다
       실패   : timeout_s 초과 / launch 프로세스 사망 / NO_DATA_S 동안 로그 무변화
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
import shutil
import signal
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'tools'))
sys.path.insert(0, str(_ROOT / 'src' / 'hlfma'))

from scp_client import ScpClient                       # noqa: E402

END_MARGIN_M = 5.0        # 완주: route_s >= total - 이 값 (summarize_run 과 동일 기준)
STOP_GRACE_S = 10.0       # 완주 도달 후 정지(v<0.5)를 기다리는 최대 시간
NO_DATA_S = 20.0          # 로그가 이 시간 동안 안 자라면 no-data 실패
POLL_S = 2.0
DONE_RULE = (f'완주 판정: 로그 ego.route_s ≥ route_{{name}}.pkl 총길이 − {END_MARGIN_M} m, '
             f'도달 후 v < 0.5 m/s 또는 {STOP_GRACE_S:.0f} s 유예')


def load_scenarios(path: str) -> list[dict]:
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
                raise SystemExit(f'시나리오 [{i}] 에 {k} 가 없다: {it}')
        it.setdefault('timeout_s', 180.0)
        out.append(it)
    names = [it['name'] for it in out]
    if len(set(names)) != len(names):
        raise SystemExit(f'시나리오 이름이 중복된다: {names}')
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


class Runner:
    def __init__(self, args) -> None:
        self.args = args
        self.launch: subprocess.Popen | None = None
        self.scp: ScpClient | None = None
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = _ROOT / 'logs' / 'batch' / ts
        self.results: list[dict] = []

    # ── 정리 (Ctrl+C 포함 모든 경로에서 호출) ─────────────────────────────
    def cleanup(self) -> None:
        if self.launch is not None and self.launch.poll() is None:
            try:
                os.killpg(self.launch.pid, signal.SIGINT)
                try:
                    self.launch.wait(timeout=10.0)
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

    # ── 한 시나리오 ───────────────────────────────────────────────────────
    def run_one(self, sc: dict) -> dict:
        name = sc['name']
        res = {'name': name, 'status': '?', 'log': None}
        route_pkl = _ROOT / 'data' / f'route_{name}.pkl'

        # 1) route 빌드 — 공용 data/route.pkl 은 건드리지 않는다
        build_cmd = [sys.executable, str(_ROOT / 'tools' / 'build_route.py'),
                     str(_ROOT / 'data' / 'lane_graph.pkl'), sc['route_csv'],
                     '-o', str(route_pkl)]
        launch_cmd = ['ros2', 'launch', 'hlfma', 'drive.launch.py',
                      f'route:={route_pkl}',
                      f'graph:={_ROOT / "data" / "lane_graph.pkl"}',
                      f'host:={self.args.host}']
        if self.args.dry_run:
            print(f'\n[dry-run] {name}')
            print('  route :', ' '.join(build_cmd))
            print('  scp   : load_and_run(%r) @ %s:48179' % (sc['vtd_xml_path'], self.args.host))
            print('  launch:', ' '.join(launch_cmd))
            print(f'  종료   : {DONE_RULE.format(name=name)} / timeout {sc["timeout_s"]}s')
            res['status'] = 'dry-run'
            return res

        cp = subprocess.run(build_cmd, capture_output=True, text=True, timeout=120)
        (self.out_dir / f'{name}.build_route.txt').write_text(cp.stdout + cp.stderr)
        if cp.returncode != 0 or not route_pkl.exists():
            res['status'] = 'route 실패'
            print(f'  ✗ build_route 실패 (rc={cp.returncode}) — {name}.build_route.txt 참고')
            return res
        import pickle
        total = float(pickle.load(open(route_pkl, 'rb'))['total_length'])
        res['route_total_m'] = round(total, 1)

        # 2) VTD 로드→Init→Start
        try:
            self.scp = ScpClient(self.args.host, verbose=False).connect()
            self.scp.load_and_run(sc['vtd_xml_path'], settle_s=self.args.settle_s)
        except OSError as e:
            res['status'] = f'SCP 실패: {e}'
            self.cleanup()
            return res

        # 3) 컨트롤러 launch (새 세션 = 프로세스 그룹 → SIGINT 로 통째로 정리)
        before = set((_ROOT / 'logs').glob('run_*.jsonl'))
        launch_out = open(self.out_dir / f'{name}.launch.txt', 'w')
        self.launch = subprocess.Popen(launch_cmd, cwd=str(_ROOT), start_new_session=True,
                                       stdout=launch_out, stderr=subprocess.STDOUT)

        # 4) 로그 감시 — 새 run_*.jsonl 이 이 시나리오의 로그다
        log_path = None
        t0 = time.monotonic()
        deadline = t0 + float(sc['timeout_s'])
        last_size, last_grow = -1, time.monotonic()
        reached_at = None
        status = 'timeout'
        while time.monotonic() < deadline:
            time.sleep(POLL_S)
            if self.launch.poll() is not None:
                status = f'launch 사망 (rc={self.launch.returncode})'
                break
            if log_path is None:
                new = set((_ROOT / 'logs').glob('run_*.jsonl')) - before
                if new:
                    log_path = max(new, key=lambda p: p.stat().st_mtime)
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
            rs = float(tick['ego']['route_s'])
            v = float(tick['ego']['speed'])
            if rs >= total - END_MARGIN_M:
                reached_at = reached_at or time.monotonic()
                if v < 0.5 or time.monotonic() - reached_at > STOP_GRACE_S:
                    status = '완주'
                    break

        # 5) 정리 + 로그 회수
        self.cleanup()
        launch_out.close()
        res['status'] = status
        if log_path is not None and log_path.exists():
            dst = self.out_dir / f'{name}.jsonl'
            shutil.copy2(log_path, dst)
            res['log'] = str(dst)
        return res

    # ── 결과 수집 ─────────────────────────────────────────────────────────
    def collect(self, res: dict) -> None:
        if not res.get('log'):
            return
        try:
            from summarize_run import load_map, summarize
            import pickle
            lg, _ = load_map()
            route = pickle.load(open(_ROOT / 'data' / f"route_{res['name']}.pkl", 'rb'))
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
            res['collect_error'] = f'{type(e).__name__}: {e}'
        try:
            cp = subprocess.run([sys.executable, str(_ROOT / 'tools' / 'score.py'), res['log']],
                                capture_output=True, text=True, timeout=120)
            (self.out_dir / f"{res['name']}.score.txt").write_text(cp.stdout + cp.stderr)
            res['score_ok'] = (cp.returncode == 0)
        except Exception as e:                          # noqa: BLE001
            res['score_ok'] = None
            res['collect_error'] = res.get('collect_error', '') + f' score:{e}'

    # ── 표 ────────────────────────────────────────────────────────────────
    def report(self) -> str:
        hdr = ['시나리오', '상태', '완주', '거리[m]', '시간[s]', '평균[km/h]',
               '최소객체거리[m]', 'E_STOP', 'shield', '과속틱', '감점없음', '비고']
        rows = []
        for r in self.results:
            col = 'Y' if r.get('min_obj_dist_m') is not None and r['min_obj_dist_m'] < 2.5 else ''
            rows.append([
                r['name'], r['status'],
                {True: 'O', False: 'X'}.get(r.get('done'), '-'),
                r.get('dist_m', '-'), r.get('time_s', '-'), r.get('avg_kph', '-'),
                ('%s%s' % (r.get('min_obj_dist_m', '-'), ' ⚠충돌?' if col else '')),
                r.get('estop_ticks', '-'),
                ','.join(f'{k}:{v}' for k, v in (r.get('shield') or {}).items()) or '-',
                r.get('overspeed_ticks', '-'),
                {True: 'O', False: 'X', None: '-'}.get(r.get('score_ok'), '-'),
                r.get('collect_error', ''),
            ])
        w = [max(len(str(h)), *(len(str(row[i])) for row in rows)) if rows else len(str(h))
             for i, h in enumerate(hdr)]
        lines = ['  '.join(str(c).ljust(w[i]) for i, c in enumerate(hdr))]
        lines += ['  '.join(str(c).ljust(w[i]) for i, c in enumerate(row)) for row in rows]
        return '\n'.join(lines)

    # ── 전체 ──────────────────────────────────────────────────────────────
    def run(self) -> int:
        scenarios = load_scenarios(self.args.scenarios)
        if not self.args.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f'배치 {len(scenarios)}건, host={self.args.host}')
        print(DONE_RULE.format(name='<name>'))
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
        print(DONE_RULE.format(name='<name>'))
        print(table)
        if not self.args.dry_run:
            (self.out_dir / 'report.txt').write_text(
                DONE_RULE.format(name='<name>') + '\n' + table + '\n', encoding='utf-8')
            (self.out_dir / 'report.json').write_text(
                json.dumps(self.results, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'\n리포트: {self.out_dir}/report.txt (.json, 개별 로그·score 전문 포함)')
        return 0 if all(r['status'] in ('완주', 'dry-run') for r in self.results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='배치 회귀 실행기 (SCP 원격 제어)')
    ap.add_argument('scenarios', help='시나리오 목록 json/yaml')
    ap.add_argument('--host', default='192.168.10.1', help='VTD 주소 (SCP 48179 + 9910)')
    ap.add_argument('--settle-s', type=float, default=3.0, help='SCP 명령 간 대기')
    ap.add_argument('--pause-s', type=float, default=3.0, help='시나리오 간 대기')
    ap.add_argument('--dry-run', action='store_true', help='실행 없이 계획만 출력')
    a = ap.parse_args(argv)
    return Runner(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
