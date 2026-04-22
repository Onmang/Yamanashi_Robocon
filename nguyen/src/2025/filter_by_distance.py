# -*- coding: utf-8 -*-
# RealSenseカメラで深度によるフィルタリング（レンジ指定）
# 依存関係: pip install opencv-python pyrealsense2

import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path

# パラメータ保存用のファイルパス
PARAM_PATH = "distance_params.json"

def _noop(x): pass

def create_distance_trackbars(win, max_dist_cm=200):
    """距離[cm]を調整するトラックバーを作成する"""
    # D405を想定した初期値 (7cm - 50cm)
    cv2.createTrackbar('Dist_min [cm]', win, 7,  max_dist_cm, _noop)
    cv2.createTrackbar('Dist_max [cm]', win, 50, max_dist_cm, _noop)

def get_distance_range(win):
    """トラックバーから距離[cm]の範囲を取得する"""
    d_min_cm = cv2.getTrackbarPos('Dist_min [cm]', win)
    d_max_cm = cv2.getTrackbarPos('Dist_max [cm]', win)
    return d_min_cm, d_max_cm

def save_params(path, d_min_cm, d_max_cm):
    """パラメータをJSONファイルに保存する"""
    data = {'Dist_min [cm]': d_min_cm, 'Dist_max [cm]': d_max_cm}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved Distance params -> {path}")

def load_params_if_exist(win, path):
    """パラメータが既存ならJSONファイルから読み込む"""
    p = Path(path)
    if not p.exists(): return
    data = json.loads(p.read_text(encoding='utf-8'))
    for k, v in data.items():
        cv2.setTrackbarPos(k, win, int(v))
    print(f"Loaded Distance params <- {path}")

def main():
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
    cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Distance Control', cv2.WINDOW_NORMAL)

    # ウィンドウが重ならないように初期位置を設定
    cv2.moveWindow('Input', 0, 0)
    cv2.moveWindow('Filtered', W, 0)
    cv2.moveWindow('Depth', 0, H + 40) # 40はウィンドウのタイトルバー分
    cv2.moveWindow('Distance Control', W, H + 40)

    # トラックバーを作成
    create_distance_trackbars('Distance Control')
    load_params_if_exist('Distance Control', PARAM_PATH)

    print("操作: s=パラメータ保存, q/ESC=終了")

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
            
            cv2.putText(color_image, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.drawMarker(color_image, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # トラックバーから距離の範囲を[cm]で取得
            dist_min_cm, dist_max_cm = get_distance_range('Distance Control')

            # dist_min が dist_max を超えた場合、値を入れ替える
            if dist_min_cm > dist_max_cm:
                dist_min_cm, dist_max_cm = dist_max_cm, dist_min_cm
                cv2.setTrackbarPos('Dist_min [cm]', 'Distance Control', dist_min_cm)
                cv2.setTrackbarPos('Dist_max [cm]', 'Distance Control', dist_max_cm)

            # 距離[cm]を、正しいdepth_scaleを使ってraw深度値に変換
            dist_min_raw = (dist_min_cm / 100.0) / depth_scale
            dist_max_raw = (dist_max_cm / 100.0) / depth_scale

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, int(dist_min_raw), int(dist_max_raw))

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            # 深度画像を可視化用にカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 各画像を表示
            cv2.imshow('Input', color_image)
            cv2.imshow('Filtered', filtered_image)
            cv2.imshow('Depth', depth_colormap)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('s'):
                save_params(PARAM_PATH, dist_min_cm, dist_max_cm)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

