#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""activate_realsense_camera.py

activate realsense camera

Example:
    $ ./<filename>.py  # after run: chmod +x <filename>.py

Author:
    nguyen

Date:
    2026-07-04

History:
    - 2026-07-04: nguyen coped from 2025
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# ストリームの設定
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

ctx = rs.context()
devices = ctx.query_devices()
print(f"[Debug] Detected {len(devices)} devices")

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    name   = dev.get_info(rs.camera_info.name)
    print(f"\n=== Camera {i} ===")
    print("Name  :", name)
    print("Serial:", serial)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

try:
    print("[Debug] Starting capturing...")
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Convert images to numpy arrays
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        # イメージの結合
        images = np.hstack((color_image, depth_colormap))

        # 表示
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        #cv2.waitKey(1)

        # q キー入力で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    # ストリーミング停止
    pipeline.stop()