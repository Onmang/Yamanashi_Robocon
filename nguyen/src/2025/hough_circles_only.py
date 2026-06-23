# -*- coding: utf-8 -*-
# ハフ変換のテスト


import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
import matplotlib.pyplot as plt
import serial
import time
import sys

PARAM_HOUGH = "houghcircles_params_test.json"

def _noop(x): pass

def create_houghcircles_trackbars(win):
    cv2.createTrackbar('minDist', win, 50, 500, _noop)
    cv2.createTrackbar('param1', win, 100, 300, _noop)
    cv2.createTrackbar('param2', win, 30, 150, _noop)
    cv2.createTrackbar('minRadius', win, 10, 100, _noop)
    cv2.createTrackbar('maxRadius', win, 100, 200, _noop)

def get_houghcircles_params(win):
    minDist   = cv2.getTrackbarPos('minDist', win)
    param1    = cv2.getTrackbarPos('param1', win)
    param2    = cv2.getTrackbarPos('param2', win)
    minRadius = cv2.getTrackbarPos('minRadius', win)
    maxRadius = cv2.getTrackbarPos('maxRadius', win)
    return minDist, param1, param2, minRadius, maxRadius

def save_params_hough(path, minDist, param1, param2, minRadius, maxRadius):
    """パラメータをJSONファイルに保存する"""
    data = {'minDist': minDist, 'param1': param1, 'param2': param2,
            'minRadius': minRadius,  'maxRadius': maxRadius}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved HoughCircles params -> {path}")

def load_params_if_exist(win, path):
    """パラメータが既存ならJSONファイルから読み込む"""
    p = Path(path)
    if not p.exists(): return
    data = json.loads(p.read_text(encoding='utf-8'))
    for k, v in data.items():
        cv2.setTrackbarPos(k, win, int(v))
    print(f"Loaded params <- {path}")

def main():

    # argv
    args = sys.argv
    if len(args) < 2:
        print("============ Error =============================================")
        print("Usage: python recognition_ver1.py [0:red, 1:yellow, 2:blue]")
        print("================================================================")
        return
    hsv_param_num = int(args[1])
    pipeline = rs.pipeline()
    cfg = rs.config()

    # 解像度とFPS
    W, H, FPS = 848, 480, 30

    # 深度とカラーのストリームを有効化
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    # ストリーミング開始
    profile = pipeline.start(cfg)
    
    # 内部パラメータの行列
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    inst_matrix = np.array([[intr.fx, 0, intr.ppx],
                            [0, intr.fy, intr.ppy],
                            [0, 0, 1]])
    print(f"Camera Intrinsics: {intr.width}x{intr.height}")
    print("Intrinsic Matrix:")
    print("[[fx, 0, ppx],")
    print(f" [0, fy, ppy],")
    print(f" [0, 0, 1]]")
    print(inst_matrix)

    # 深度センサーの情報を取得
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # 深度センサーからdepth_scaleを取得（単位をメートルに変換する係数）
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    # 深度とカラーの位置合わせ（Align）オブジェクトを作成
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ウィンドウの作成と配置
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HoughCircles result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HoughCircles Control', cv2.WINDOW_NORMAL)


    # トラックバーを作成
    create_houghcircles_trackbars('HoughCircles Control')
    load_params_if_exist('HoughCircles Control', PARAM_HOUGH)



    # ウィンドウが重ならないように初期位置を設定
    win_w, win_h = 450, 350  # ウィンドウサイズを小さく調整
    offset_x = 50
    offset_y = 50
    cv2.moveWindow('Input', offset_x, offset_y)
    cv2.moveWindow('HoughCircles', win_w + offset_x, offset_y)
    cv2.moveWindow('HoughCircles Control', win_w + offset_x, offset_y)

    print("[Operation]: s->Save params, q/ESC->Exit")


    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 画面中央の平均距離を計算して表示
            cx, cy = W // 2, H // 2
            roi_size = 10
            x1, x2 = cx - roi_size // 2, cx + roi_size // 2
            y1, y2 = cy - roi_size // 2, cy + roi_size // 2
            
            # 中心領域
            center_roi = depth_image[y1:y2, x1:x2]

            # 0を除く平均値を計算
            non_zero_values = center_roi[center_roi > 0]
            avg_dist_raw = np.mean(non_zero_values) if non_zero_values.size > 0 else 0
            
            # 正しいdepth_scaleを使ってメートル[m]に変換
            center_dist_m = avg_dist_raw * depth_scale
            dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m*1000:.0f} [mm])"
            
            # オリジナルに載せない
            overlay = color_image.copy()
            cv2.putText(overlay, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # グレースケール変換
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # ハフ変換、円
            minDist, param1, param2, minRadius, maxRadius = get_houghcircles_params('HoughCircles Control')
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            if circles is not None:
                circles = np.uint16(np.around(circles)) 
                for i in circles[0, :]:
                    # 円の外周
                    cv2.circle(overlay, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 円の中心
                    cv2.circle(overlay, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # 円中心のdepthデータ取得
                    x, y = int(i[0]), int(i[1])
                    if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                        depth_value = depth_image[y, x]
                        depth_m = depth_value * depth_scale
                        depth_text = f"{depth_m:.3f}m"
                    else:
                        depth_text = "Depth: N/A"
                    # 中心座標と半径・depthを表示
                    cv2.putText(overlay, depth_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 各画像を表示
            cv2.imshow('Input', color_image)
            cv2.imshow('HoughCircles result', overlay)


            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('s'):
                save_params_hough(PARAM_HOUGH, minDist, param1, param2, minRadius, maxRadius)

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # 0:赤, 1:黄, 2:青

