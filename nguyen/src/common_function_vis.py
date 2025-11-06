# -*- coding: utf-8 -*-
# 共通関数, 主に描画用

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs 

from common_function import (
    compute_center_distance,
)

# 中心点描画
def draw_center_distance_debug(color_image, depth_image, cam_d435i, W=640, H=480):
    # 中心座標
        cx, cy = W // 2, H // 2
        center_dist_m, center_dist_mm, _ ,(x1, y1), (x2, y2) = compute_center_distance(
                    depth_image,
                    cam_d435i.depth_scale,  # ← d435iでもd405でもOK
                    W,
                    H,
                    cx,
                    cy,
                    roi_size=20,
                )
        dist_text = f"Center: {center_dist_m:.3f} [m] ({center_dist_mm:.0f} [mm])"

        # "input" ウィンドウ表示
        overlay = color_image.copy()
        cv2.putText(
                    overlay,
                    dist_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
        # 中心領域を矩形で表示
        cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        color=(0, 200, 0),  # 緑枠
                        thickness=1,
                    )
        return overlay