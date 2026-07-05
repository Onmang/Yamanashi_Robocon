#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_20260704_move.py

test for move 
trace ball
"recognition_ver3.py" base

Example:
    $ ./<filename>.py  # after run: chmod +x <filename>.py

Author:
    nguyen

Date:
    2026-07-04

History:
    - 2026-07-04: nguyen coped from 2025
"""
import time

import cv2
import numpy as np
import serial

from rgbd_utils import (
    init_realsense_camera
)

from path_config import (
    PARAM_PATH_DIS_LIMIT
)

# arduino シリアル通信設定
ARDUINO = False  # True: シリアル通信ON, False: シリアル通信OFF
if ARDUINO:
    global ser
    serial_port = "/dev/ttyACM0"  # arduino UNO
    
    baud_rate = 115200  # 9600, 115200
    ser = serial.Serial(
        serial_port,
        baud_rate,  # できれば 115200 を推奨
        timeout=0,  # 読み取りは非ブロッキング（読みはしてないが安全）
        write_timeout=0,  # 書き込みもブロッキングしない
    )
    time.sleep(2.0)  # リセット待ち 単位：sec
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("[Debug] Serial Port was opened:", serial_port)


def main():
    """
    flowchart
    detect ball 
    move (2025 code)

    """
    activate_cam = None 
    try:

        # --------------------------------------------
        # カメラ初期化
        # --------------------------------------------
        # 解像度とFPS
        W, H, FPS = 640, 480, 30

        # RealSense カメラ初期化
        # activate_cam = init_realsense_camera(
        #     name="d435i",
        #     serial="949122070535",  # 実機のシリアル
        #     width=W,
        #     height=H,
        #     fps=FPS,
        #     extrinsic_guess={
        #         "tx": -(32.5 * 0.001),  # m
        #         "ty": -50 * 0.001,  # m
        #         "tz": 200 * 0.001,  # m
        #         "rx_deg": -90,
        #         "ry_deg": 0,
        #         "rz_deg": 0,
        #     },
        #     dis_param_path=PARAM_PATH_DIS_LIMIT,
        # )

        # for debug
        activate_cam = init_realsense_camera(
            name="d405",
            serial="218622274519",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": -(32.5 * 0.001),  # m
                "ty": -50 * 0.001,  # m
                "tz": 200 * 0.001,  # m
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
            dis_param_path=PARAM_PATH_DIS_LIMIT,
        )

        # init activate cam
        time.sleep(1.0)  # カメラ安定化待ち
        print("[DEBUG] Camera initialized.")

    except KeyboardInterrupt:
            print("\n===== Keyboard Interrupt =====")
            print("[FINISHED].")
            print("==============================\n")
    finally:
        if ARDUINO and ser is not None:
            ser.close()
        if activate_cam is not None:
            activate_cam.pipeline.stop()


if __name__ == "__main__":
    main()
