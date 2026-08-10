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

# from path_config import (
#     YOLO_MODEL
# )

from rgbd_utils import  find_vertical_edge


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

# 赤色 HSV 範囲(赤は色相環の両端にまたがるので2レンジ)
RED_HSV_LOWER_1 = (0, 80, 50)
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 80, 50)
RED_HSV_UPPER_2 = (180, 255, 255)


def expand_bbox_ver2(bbox, frame_shape, margin_ratio=0.3):
    """bboxをmargin_ratio分だけ外側に広げ、画像範囲内にクリップする"""
    x1, y1, x2, y2 = map(int, bbox)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    h, w = frame_shape[:2]
    x1 = max(0, int(x1 - margin_ratio * dx))
    x2 = min(w, int(x2 + margin_ratio * dx))
    y1 = max(0, int(y1 - margin_ratio * dy))
    y2 = min(h, int(y2 + margin_ratio * dy))
    return x1, y1, x2, y2

def expand_bbox(bbox, frame_shape, margin_ratio=0.3):
    """bboxをmargin_ratio分だけ外側に広げ、画像範囲内にクリップする"""
    x1, y1, x2, y2 = map(int, bbox)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dis = np.sqrt(dx**2+dy**2)
    h, w = frame_shape[:2]
    x1 = max(0, int(x1 - margin_ratio * dis))
    x2 = min(w, int(x2 + margin_ratio * dis))
    y1 = max(0, int(y1 - margin_ratio * dis))
    y2 = min(h, int(y2 + margin_ratio * dis))
    return x1, y1, x2, y2

def detect_triangle_by_color(frame, bbox, epsilon_ratio=0.03, area_min=200, margin_ratio=0.3):
    """
    bbox内を赤色でマスクし、三角形(旗)を検出する。
    detect_triangle_in_bbox(rgbd_utils.py)はCannyエッジベースだが、
    旗の下辺や先端が背景とのコントラスト不足でエッジが繋がらず検出に失敗するため、
    赤色そのものを塗りつぶしマスクとして使う方式に切り替えた。

    Args:
        frame (np.ndarray): 元画像(BGR)
        bbox (tuple): (x1, y1, x2, y2)
        epsilon_ratio (float): 近似精度
        area_min (float): 最小面積
    Returns:
        approx (np.ndarray | None): 三角形輪郭(N×1×2)。見つからなければNone
    """
    x1, y1, x2, y2 = expand_bbox(bbox, frame.shape, margin_ratio)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1) | \
           cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    best_local = None
    for cnt in contours:
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * arclen, True)
        area = cv2.contourArea(approx)
        if len(approx) == 3 and area >= area_min and area > best_area:
            best_local = approx.copy()
            approx[:, 0, 0] += x1
            approx[:, 0, 1] += y1
            best = approx
            best_area = area

    # デバッグ用: 赤マスク画像に輪郭を描画して表示
    mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_debug, contours, -1, (255, 0, 0), 1)  # 全輪郭(青)
    if best_local is not None:
        cv2.polylines(mask_debug, [best_local], isClosed=True, color=(0, 255, 255), thickness=2)  # 採用した三角形(黄)
    cv2.namedWindow("red mask (debug)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("red mask (debug)", 480, 360)
    cv2.imshow("red mask (debug)", mask_debug)

    return best


def main():
    # yolo paramas
    conf = 0.5 # 信頼度閾値 (0~1)
    iou = 0.45 # IoU閾値

    # モデル読み込み

    # lab pc
    YOLO_MODEL = "/home/konfi/Yamanashi_Robocon/nguyen/src/yolo_model/detect_5class_v8n/weights/best.onnx"
    print(f"[INFO] Starting to load YOLO model: {YOLO_MODEL}")
    s_time = time.time()
    model = YOLO(YOLO_MODEL, task='detect')
    e_time = time.time()
    run_time = e_time - s_time
    print(f"[Debug] Run time for loading YOLO model: {run_time:.3f} sec.")

    # make image
    color_image = cv2.imread("flag_3.jpg")
    # cv2.imshow("orin", color_image)


    # YOLOv8 推論
    # ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
    results = model.predict(
                    source=color_image,
                    classes=[flag_idx],
                    conf=conf,
                    iou=iou,
                    verbose=True,
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
        cls_id_ = int(box.cls[0]) #ata/src/lena.jpg class ID
        conf_ = float(box.conf[0]) # conf.

        if cls_id_== flag_idx and conf_ > max_flag_conf:
            max_flag_conf = conf_
            best_flag_box = box

    if best_flag_box is not None:
        triangles = detect_triangle_by_color(color_image, best_flag_box.xyxy[0], epsilon_ratio=0.03, area_min=AREA_MIN_FLAG, margin_ratio=0.06)
        # バウンディングボックス描画
        x1, y1, x2, y2 = expand_bbox(best_flag_box.xyxy[0], color_image.shape,margin_ratio=0.06)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bbox_color, 5)

        if triangles is not None:
            cv2.polylines(annotated_image, [triangles], isClosed=True, color=(0, 255, 255), thickness=2)

            edge = find_vertical_edge(triangles)
            if edge is not None:
                p1, p2 = edge
                mx = int((p1[0] + p2[0]) / 2)
                my = int((p1[1] + p2[1]) / 2)
                cv2.line(annotated_image, tuple(p1), tuple(p2), (0, 0, 255), 3)
                cv2.drawMarker(annotated_image, position=(mx,my), color=(0,255,0),markerType=cv2.MARKER_CROSS, markerSize=100, thickness=5, line_type=cv2.LINE_4)

    # カラー画像表示
    cv2.namedWindow("YOLOv8 + RealSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 + RealSense", 640, 480)
    cv2.imshow("YOLOv8 + RealSense", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
