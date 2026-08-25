"""테스트에서 vtd_adapter / tools 를 import 할 수 있게 경로를 잡는다."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

PARAMS_YAML = str(ROOT / 'config' / 'params.yaml')
