"""테스트에서 core/ 를 ROS 빌드 없이 import 할 수 있게 경로를 잡는다."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / 'src' / 'hlfma'
for p in (str(ROOT), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

PARAMS_YAML = str(PKG / 'config' / 'params.yaml')
