import os


_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(_ENV_DIR, "assets")


DIGITAL_TWIN_BACKGROUND_IMAGES = {
    "3rdview_camera": os.path.join(
        _ASSET_DIR, "backgrounds", "thirdview_background.png"
    ),
}


PANDA_UMI_REFERENCE_QPOS = [
    0.0,
    0.39269908169872414,
    0.0,
    -1.9634954084936207,
    0.0,
    2.356194490192345,
    0.7853981633974483,
    0.04,
    0.04,
]

PANDA_UMI_CAMERA_POSES = {
    "3rdview_camera": {
        "mount_link": "panda_link0",
        "p": [1.028399986258626, 0.03181203390946734, 0.5650221695897678],
        "q": [0.017373342776650636, -0.4402130067970881, 0.01681990116140459, 0.8975676946795448],
    },
    "hand_camera": {
        "mount_link": "camera_link",
        "p": [-0.004933241890954809, 0.019370738865599802, 0.006932944573748325],
        "q": [0.9997672000023046, 0.0021878508838338684, -0.015248489267897954, -0.01510770277405003],
    },
}

PANDA_UMI_CAMERA_SETTINGS = {
    "3rdview_camera": {
        "width": 640,
        "height": 480,
        "near": 0.01,
        "far": 100.0,
        "fov_deg": 44.0,
    },
    "hand_camera": {
        "width": 640,
        "height": 480,
        "near": 0.01,
        "far": 100.0,
        "fov_deg": 44.0,
    },
}

from rlinf.envs.maniskill.tasks.digital_twin.digital_twin_based_env import (
    DigitalTwinBaseEnv,
)
from rlinf.envs.maniskill.tasks.digital_twin.push_button import PushButtonEnv
from rlinf.envs.maniskill.tasks.digital_twin.push_button_dr import (
    PushButtonDomainRandomizedEnv,
)

__all__ = [
    "DIGITAL_TWIN_BACKGROUND_IMAGES",
    "DigitalTwinBaseEnv",
    "PANDA_UMI_CAMERA_POSES",
    "PANDA_UMI_CAMERA_SETTINGS",
    "PANDA_UMI_REFERENCE_QPOS",
    "PushButtonEnv",
    "PushButtonDomainRandomizedEnv",
]
