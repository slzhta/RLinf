# Copyright 2025 The RLinf Authors.
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

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():

    serial_numbers = []
    for device in rs.context().devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Found camera: {serial_number}")
        serial_numbers.append(serial_number)

    if not serial_numbers:
        raise RuntimeError("No RealSense camera found.")

    pipelines = {}
    for serial_number in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            15,
        )
        profile = pipeline.start(config)
        
        device = profile.get_device()
        color_sensor = device.query_sensors()[1]

        if color_sensor.supports(rs.option.saturation):
            color_sensor.set_option(rs.option.saturation, 85)
        
        if color_sensor.supports(rs.option.contrast):
            color_sensor.set_option(rs.option.contrast, 60)


        pipelines[serial_number] = pipeline

    for pipeline in pipelines.values():
        pipeline.stop()


if __name__ == "__main__":
    main()