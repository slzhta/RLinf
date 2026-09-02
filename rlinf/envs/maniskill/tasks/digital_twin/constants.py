# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
DIGITAL_TWIN_ASSET_DIR = _ENV_DIR / "assets"

DIGITAL_TWIN_BACKGROUND_IMAGES = {
    "3rdview_camera": str(
        DIGITAL_TWIN_ASSET_DIR / "backgrounds" / "thirdview_background.png"
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
        "q": [
            0.017373342776650636,
            -0.4402130067970881,
            0.01681990116140459,
            0.8975676946795448,
        ],
    },
    "hand_camera": {
        "mount_link": "camera_link",
        "p": [
            -0.004933241890954809,
            0.019370738865599802,
            0.006932944573748325,
        ],
        "q": [
            0.9997672000023046,
            0.0021878508838338684,
            -0.015248489267897954,
            -0.01510770277405003,
        ],
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
