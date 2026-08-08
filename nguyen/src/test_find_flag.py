#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_find_flag.py

test for moving and hitting
trace ball
"recognition_ver3.py" base

Example:
    $ ./<filename>.py  # after run: chmod +x <filename>.py

            # time measurement
            start_time = time.time()

            # end
            end_time = time.time()
            runtime = start_time-end_time
            print(f"[time]{runtime}")

Author:
    nguyen

Date:
    2026-08-05

History:
    - 2026-08-05: nguyen 
"""
import time
import numpy as np
import torch
import cv2
from ultralytics import YOLO

from rgbd_utils import (
    detect_triangle_in_bbox
)

from path_config import (
    YOLO_MODEL
)

# macro
DETECTION = "detection"
ARDUINO_MOVE = "arduino_move"
HIT = "hit"
BALL = "ball"
FLAG = "flag"
POLE = "pole"
# ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
ball_idx = 4
flag_idx = 1
pole_idx = 2
bbox_color = (0, 255, 0) # green,  バウンディングボックス描画

# しきい値関係
AREA_MIN = 100  # 小ノイズ除去
AREA_MIN_FLAG = 120  # flag用三角形最小面積

# threshold
dist_th_1 = 100  # mm
angle_th_1 = 1  # deg


def main():
    # yolo paramas
    conf = 0.5 # 信頼度閾値 (0~1)
    iou = 0.45 # IoU閾値

    # モデル読み込み
    print(f"[INFO] Starting to load YOLO model: {YOLO_MODEL}")
    s_time = time.time()
    model = YOLO(YOLO_MODEL, task='detect')
    e_time = time.time()
    run_time = e_time - s_time
    print(f"[Debug] Run time for loading YOLO model: {run_time:.3f} sec.")

    # make image
    color_image = cv2.imread('data/src/lena.jpg')

    # YOLOv8 推論
    # ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
    results = model.predict(
                    source=color_image,
                    #classes=[ball_idx, flag_idx, pole_idx],
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )
    result = results[0]  # this is List type

    # FPS 表示
    annotated_image = color_image.copy()
    fps_text = f"FPS: {1 / (results[0].speed['inference'] / 1000 + 1e-9):.1f}"
    cv2.putText(
                    annotated_image, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

    # best ball
    best_flag_box = None
    max_flag_conf = 0.0

    # check the all boxes
    for box in result.boxes:
        cls_id_ = int(box.cls[0]) # class ID
        conf_ = float(box.conf[0]) # conf.

        if cls_id_== flag_idx and conf_ > max_flag_conf:
            max_flag_conf = conf_
            best_flag_box = box

        triangles = detect_triangle_in_bbox(color_image, best_flag_box.xyxy[0], epsilon_ratio=0.08, area_min=AREA_MIN_FLAG)
        if triangles is not None:
            cv2.polylines(annotated_image, [triangles], isClosed=True, color=(0, 255, 255), thickness=2)

    # カラー画像表示
    cv2.imshow("YOLOv8 + RealSense", annotated_image)


if __name__ == "__main__":
    main()
