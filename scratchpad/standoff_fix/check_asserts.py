"""pytest 미설치 환경 — shift_latest_m/standoff_floor_m 에 걸린 기존 assert 를 직접 확인."""
import sys, math, pathlib
ROOT=pathlib.Path('/home/cjw/mando'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'team_code'))
from vtd_adapter.config import load_params_yaml
CFG=load_params_yaml(str(ROOT/'config'/'params.yaml'))
OT=CFG['overtake']; A=CFG['speed']['stop_profile_a']
S=OT['shift_latest_m']; F=OT.get('standoff_floor_m')
checks=[
 ("test_side_pass.py:52   STANDOFF == 25.0 and standoff_stop_s == 1.0", S==25.0 and OT['standoff_stop_s']==1.0),
 ("test_standoff_chain.py:45  blocker_dist_max >= STANDOFF+5", OT['blocker_dist_max']>=S+5.0),
 ("test_avoid.py:278      max(shift_latest_m,k*v) >= shift_latest_m", max(S,OT['shift_k_s']*13.9)>=S),
]
for n,ok in checks: print(('  ✅' if ok else '  ✗ ')+f" {n}")
print(f"\n  shift_latest_m = {S}   standoff_floor_m = {F}   (기본 동일 = 무변화)")
