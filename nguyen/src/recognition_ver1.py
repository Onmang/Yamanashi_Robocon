# -*- coding: utf-8 -*-
# 認識バージョン１


import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
import matplotlib.pyplot as plt
import serial
import time

# パラメータ保存用のファイルパス
PARAM_PATH_DIS = "distance_params.json"
PARAM_PATH_HSV = "hsv_params.json"


# ls -l /dev/ | grep tty
# Arduinoが接続されているシリアルポートとボーレートを設定
# serial_port = '/dev/ttyACM0'
serial_port = '/dev/ttyUSB0'

baud_rate = 115200
ser = None


def _noop(x): pass

def create_hsv_trackbars(win):
    cv2.createTrackbar('H_low',  win, 0,   179, _noop)
    cv2.createTrackbar('H_high', win, 179, 179, _noop)
    cv2.createTrackbar('S_low',  win, 0,   255, _noop)
    cv2.createTrackbar('S_high', win, 255, 255, _noop)
    cv2.createTrackbar('V_low',  win, 0,   255, _noop)
    cv2.createTrackbar('V_high', win, 255, 255, _noop)

def get_hsv_range(win):
    hl = cv2.getTrackbarPos('H_low',  win)
    hh = cv2.getTrackbarPos('H_high', win)
    sl = cv2.getTrackbarPos('S_low',  win)
    sh = cv2.getTrackbarPos('S_high', win)
    vl = cv2.getTrackbarPos('V_low',  win)
    vh = cv2.getTrackbarPos('V_high', win)
    return (hl, sl, vl), (hh, sh, vh)

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

def save_params_dis(path, d_min_cm, d_max_cm):
    """パラメータをJSONファイルに保存する"""
    data = {'Dist_min [cm]': d_min_cm, 'Dist_max [cm]': d_max_cm}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved Distance params -> {path}")

def save_params_hsv(path, lo, hi):
    data = {'H_low':lo[0],'S_low':lo[1],'V_low':lo[2],
            'H_high':hi[0],'S_high':hi[1],'V_high':hi[2]}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved HSV params -> {path}")

def load_params_if_exist(win, path):
    """パラメータが既存ならJSONファイルから読み込む"""
    p = Path(path)
    if not p.exists(): return
    data = json.loads(p.read_text(encoding='utf-8'))
    for k, v in data.items():
        cv2.setTrackbarPos(k, win, int(v))
    print(f"Loaded params <- {path}")

def show_hist(img_hsv):
    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    plt.figure(figsize=(12, 8))

    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()

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
    cv2.namedWindow('Depth Filter', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Distance Control', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Control', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Mask Morph', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    # トラックバーを作成
    create_distance_trackbars('Distance Control')
    load_params_if_exist('Distance Control', PARAM_PATH_DIS)
    create_hsv_trackbars('HSV Control')
    load_params_if_exist('HSV Control', PARAM_PATH_HSV)

    # ウィンドウが重ならないように初期位置を設定
    win_w, win_h = 450, 400  # ウィンドウサイズを小さく調整
    offset_x = 50
    offset_y = 45
    cv2.moveWindow('Input', offset_x, offset_y)
    cv2.moveWindow('Depth Filter', win_w + offset_x, offset_y)
    cv2.moveWindow('Depth', 2 * win_w + offset_x, offset_y)
    cv2.moveWindow('HSV Mask', offset_x, win_h + offset_y)
    cv2.moveWindow('HSV Mask Morph', win_w + offset_x, win_h + offset_y)
    cv2.moveWindow('Result', 2 * win_w + offset_x, win_h + offset_y)

    cv2.moveWindow('Distance Control', 3 * win_w + offset_x, offset_y)
    cv2.moveWindow('HSV Control', 3 * win_w + offset_x, win_h + offset_y)

    print("[Operation]: s->Save params, q/ESC->Exit")

    # arduino　送信設定
    # シリアルポートを開く
    ser = serial.Serial(serial_port, baud_rate)
    print("Serial Port was opened: " + serial_port)
    mode = 1
    angle = 0
    dis = 0

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

            # center_dist_m を dis（mm, 整数）に格納してArduinoへ送信
            dis_mm = int(center_dist_m * 1000) if center_dist_m > 0 else 0
            dis = dis_mm % 10000  # 0..9999 に収める

            # Format with zero-padding to fixed widths so Arduino can parse consistently
            angle = angle % 1000       # 0..999
            msg = f"{mode}{angle:03d}{dis:04d}\n"

            try:
                ser.write(msg.encode('ascii'))
                print(f"Sent: {msg.strip()}")
            except Exception as e:
                print("Failed to write to serial: " + str(e))
            
            cv2.putText(color_image, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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

            # 低ノイズ化したいときは有効化
            filtered_image = cv2.GaussianBlur(filtered_image, (5,5), 0)

            # HSV変換して色抽出
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            lo, hi = get_hsv_range('HSV Control')
            hsv_mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

            # 膨張・収縮でマスク整形（任意）
            kernel = np.ones((3,3), np.uint8)
            mask_morph = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3)

            # 可視化（マスクをカラーに適用）
            vis = cv2.bitwise_and(filtered_image, filtered_image, mask=mask_morph)

            # ハフ変換、円
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                       param1=100, param2=30, minRadius=10, maxRadius=100)
            if circles is not None:
                circles = np.uint16(np.around(circles)) 
                for i in circles[0, :]:
                    # 円の外周
                    cv2.circle(vis, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 円の中心
                    cv2.circle(vis, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # 円中心のdepthデータ取得
                    x, y = int(i[0]), int(i[1])
                    if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                        depth_value = depth_image[y, x]
                        depth_m = depth_value * depth_scale
                        depth_text = f"{depth_m:.3f}m"
                    else:
                        depth_text = "Depth: N/A"
                    # 中心座標と半径・depthを表示
                    cv2.putText(vis, depth_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 各画像を表示
            cv2.imshow('Input', color_image)
            cv2.imshow('Depth Filter', filtered_image)
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('HSV Mask', hsv_mask)
            cv2.imshow('HSV Mask Morph', mask_morph)
            cv2.imshow('Result', vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                ser.close()
                break
            elif k == ord('s'):
                save_params_dis(PARAM_PATH_DIS, dist_min_cm, dist_max_cm)
                save_params_hsv(PARAM_PATH_HSV, lo, hi)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

