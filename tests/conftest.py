"""테스트에서 core/ 를 ROS 빌드 없이 import 할 수 있게 경로를 잡는다."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / 'src' / 'hlfma'
for p in (str(ROOT), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

PARAMS_YAML = str(PKG / 'config' / 'params.yaml')


def pytest_configure(config):
    """
    ROS(jazzy) 의 launch_testing pytest 플러그인이 붙어 있으면 `pytest tests` 처럼
    **디렉터리** 를 주었을 때 아무것도 수집하지 않는다 (파일을 직접 주면 된다).
    ini 의 `-p no:` 는 엔트리포인트 플러그인이 이미 로드된 뒤라 듣지 않으므로
    여기서 떼어 낸다. core 테스트는 ROS 와 무관하다.
    """
    pm = config.pluginmanager
    for name in ('launch_testing', 'launch_testing_ros_pytest_entrypoint'):
        p = pm.get_plugin(name)
        if p is not None:
            pm.unregister(p)
