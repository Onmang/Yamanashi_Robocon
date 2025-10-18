# -*- coding: utf-8 -*-
# 認識バージョン１
# パラメータ関係はd435iベースにやっている


import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
import matplotlib.pyplot as plt
import serial
import time
import sys

from common_function import create_homogeneous_matrix, project_center_to_robot, CameraParam

# パラメータ保存用のファイルパス
PARAM_PATH_DIS = "distance_params.json"
PARAM_PATH_HSV = ["hsv_params_red.json", "hsv_params_yellow.json", "hsv_params_blue.json", "hsv_params_flag.json"] # 保存先パス選択
PARAM_HOUGH = "houghcircles_params.json"
PARAM_FILTER = "filter_params.json"  # ノイズフィルタGUIの保存先

# --- 円形度ベースの円検出（Contours + Circularity） ---
# 円形度 C = 4πA / P^2 （A: 面積, P: 周長）
# 目安: 完全な円で 1.0、楕円/いびつ形で低下。0.80〜0.90 くらいが実用。
CIRC_MIN = 0.80
AREA_MIN = 100       # 小ノイズ除去
AREA_MAX = 10000     # 大きすぎる塊を除外（必要に応じ調整）


# Arduino接続設定
ARDUINO = False
if ARDUINO:
    # ls -l /dev/ | grep tty
    # Arduinoが接続されているシリアルポートとボーレートを設定
    # serial_port = '/dev/ttyACM0'        # arduino UNO
    # serial_port = '/dev/ttyUSB0'      # nakano arduino mega
    serial_port = '/dev/ttyACM0'        # takemichi arduino nano evry
    baud_rate = 9600   # 9600, 115200
    ser = None

def _noop(x): pass

def create_noise_trackbars(win):
    # 0: None, 1: Median, 2: Gaussian
    cv2.createTrackbar("filter_type (0:none 1:median 2:gauss)", win, 2, 2, _noop)
    cv2.createTrackbar("ksize (odd)", win, 3, 10, _noop)  # 実効は 2*val+1 → 5,7,...
    cv2.createTrackbar("sigmaX (gauss)", win, 0, 50, _noop)  # 0なら自動

def get_noise_params(win):
    ftype = cv2.getTrackbarPos("filter_type (0:none 1:median 2:gauss)", win)
    ksize_raw = cv2.getTrackbarPos("ksize (odd)", win)
    k = max(3, 2 * ksize_raw + 1)  # 3,5,7,9,...（最低3）
    sigmaX = cv2.getTrackbarPos("sigmaX (gauss)", win)
    return ftype, k, sigmaX

def save_noise_params(path, ftype, k, sigmaX):
    """パラメータをJSONファイルに保存する"""
    data = {'filter_type': ftype, 'ksize': k, 'sigmaX': sigmaX}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved Filter params -> {path}")

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

def create_distance_trackbars(win, max_dist_cm=400):
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

    # argv
    args = sys.argv
    if len(args) < 2:
        print("============ Error =============================================")
        print("Usage: python recognition_ver1.py [0:red, 1:yellow, 2:blue]")
        print("================================================================")
        return
    hsv_param_num = int(args[1])

    # 同時変換行列
    deg = np.deg2rad  # ← 関数オブジェクトを代入
    T_cam2rob = create_homogeneous_matrix(
    tx=0, ty=0, tz=0.011,
    rx=deg(-90), ry=deg(0), rz=deg(0)
    )

    # カメラクラス
    rs_d435i = CameraParam()

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
    # クラス格納
    rs_d435i.intr = intr
    rs_d435i.add_ins_param(intr)
    inst_matrix = np.array([[rs_d435i.fx, 0, rs_d435i.ppx],
                            [0, rs_d435i.fy, rs_d435i.ppy],
                            [0, 0, 1]])
    print(f"Camera Intrinsics: {rs_d435i.intr.width}x{rs_d435i.intr.height}")
    print("Intrinsic Matrix:")
    print("[[fx, 0, ppx],")
    print(f" [0, fy, ppy],")
    print(f" [0, 0, 1]]")
    print(inst_matrix)

    # 深度センサーの情報を取得
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # 深度センサーからdepth_scaleを取得（単位をメートルに変換する係数）
    rs_d435i.depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {rs_d435i.depth_scale}")

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
    cv2.namedWindow('HoughCircles Control', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Filter GUI', cv2.WINDOW_NORMAL)

    # トラックバーを作成
    create_distance_trackbars('Distance Control')
    load_params_if_exist('Distance Control', PARAM_PATH_DIS)
    create_hsv_trackbars('HSV Control')
    load_params_if_exist('HSV Control', PARAM_PATH_HSV[hsv_param_num])
    create_houghcircles_trackbars('HoughCircles Control')
    load_params_if_exist('HoughCircles Control', PARAM_HOUGH)
    create_noise_trackbars('Filter GUI')
    load_params_if_exist('Filter GUI', PARAM_FILTER)


    # ウィンドウが重ならないように初期位置を設定
    win_w, win_h = 450, 350  # ウィンドウサイズを小さく調整
    offset_x = 50
    offset_y = 50
    cv2.moveWindow('Input', offset_x, offset_y)
    cv2.moveWindow('Depth Filter', win_w + offset_x, offset_y)
    cv2.moveWindow('Depth', 2 * win_w + offset_x, offset_y)
    cv2.moveWindow('HSV Mask', offset_x, win_h + offset_y)
    cv2.moveWindow('HSV Mask Morph', win_w + offset_x, win_h + offset_y)
    cv2.moveWindow('Result', 2 * win_w + offset_x, win_h + offset_y)

    cv2.moveWindow('Distance Control', 3 * win_w + offset_x, offset_y)
    cv2.moveWindow('HSV Control', 3 * win_w + offset_x, win_h + offset_y)
    cv2.moveWindow('HoughCircles Control', 3 * win_w + offset_x, 2 * win_h + offset_y)
    cv2.moveWindow('Filter GUI', 2 * win_w + offset_x, 2 * win_h + offset_y)

    print("[Operation]: s->Save params, q/ESC->Exit")

    # arduino　送信設定
    # シリアルポートを開く
    if ARDUINO:
        global ser
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
            center_dist_m = avg_dist_raw * rs_d435i.depth_scale
            dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m*1000:.0f} [mm])"

            # center_dist_m を dis（mm, 整数）に格納してArduinoへ送信
            if (ARDUINO):
                dis_mm = int(center_dist_m * 1000) if center_dist_m > 0 else 0
                dis = dis_mm % 10000  # 0..9999 に収める

                # Format with zero-padding to fixed widths so Arduino can parse consistently
                angle = angle % 1000       # 0..999
                msg = f"{mode}{angle:03d}{dis:04d}\n"

                try:
                    ser.write(msg.encode('ascii'))
                    time.sleep(1)
                    print(f"Sent: {msg.strip()}")
                except Exception as e:
                    print("Failed to write to serial: " + str(e))
            
            # オリジナルに載せない
            overlay = color_image.copy()
            cv2.putText(overlay, dist_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # トラックバーから距離の範囲を[cm]で取得
            dist_min_cm, dist_max_cm = get_distance_range('Distance Control')

            # dist_min が dist_max を超えた場合、値を入れ替える
            if dist_min_cm > dist_max_cm:
                dist_min_cm, dist_max_cm = dist_max_cm, dist_min_cm
                cv2.setTrackbarPos('Dist_min [cm]', 'Distance Control', dist_min_cm)
                cv2.setTrackbarPos('Dist_max [cm]', 'Distance Control', dist_max_cm)

            # 距離[cm]を、正しいdepth_scaleを使ってraw深度値に変換
            dist_min_raw = (dist_min_cm / 100.0) / rs_d435i.depth_scale
            dist_max_raw = (dist_max_cm / 100.0) / rs_d435i.depth_scale

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, int(dist_min_raw), int(dist_max_raw))

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            # 深度画像を可視化用にカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 低ノイズ化したいときは有効化
            # 低ノイズ化：GUIの選択で適用
            ftype, k, sigmaX = get_noise_params('Filter GUI')
            if ftype == 1:
                filtered_image = cv2.medianBlur(filtered_image, k)
            elif ftype == 2:
                filtered_image = cv2.GaussianBlur(filtered_image, (k, k), sigmaX if sigmaX > 0 else 0)
            # ftype == 0 は何もしない


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

            # グレースケール変換
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

            # ラベリング処理    
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)
            mask_morph_copy = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
            for i in range(1, retval):  # 0は背景なのでスキップ
                x, y, w, h, area = stats[i]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                if area >= 100 and area < 6000:  # 面積が小さいノイズを除去
                    cv2.rectangle(mask_morph_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(mask_morph_copy, (cx, cy), 3, (0, 255, 255), -1)
                    # # 中心座標のdepthデータ取得
                    # if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                    #     depth_value = depth_image[cy, cx]
                    #     depth_m = depth_value * depth_scale
                    #     depth_text = f"{depth_m:.3f}m"
                    # else:
                    #     depth_text = "Depth: N/A"
                    # # 中心座標とdepthを表示
                    # cv2.putText(mask_morph_copy, depth_text, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(mask_morph_copy, f"[{i}]:{area}", (cx + 25, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if False:
                # 2値マスクから輪郭抽出, 円を作る
                contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < AREA_MIN or area > AREA_MAX:
                        continue

                    peri = cv2.arcLength(cnt, True)
                    if peri <= 0:
                        continue

                    circularity = (4.0 * np.pi * area) / (peri * peri)
                    if circularity < CIRC_MIN:
                        continue

                    # 円近似：最小外接円（高速で安定）
                    (cx_f, cy_f), r_f = cv2.minEnclosingCircle(cnt)
                    cx, cy, r = int(cx_f), int(cy_f), int(r_f)

                    # 深度表示（任意）
                    if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                        depth_m = depth_image[cy, cx] * rs_d435i.depth_scale
                        depth_text = f"{depth_m:.3f}m"
                    else:
                        depth_text = "Depth: N/A"

                    # 可視化：マゼンタ円＋シアン中心（Hough=緑と区別）
                    cv2.circle(vis, (cx, cy), r, (255, 0, 255), 2)      # magenta
                    cv2.circle(vis, (cx, cy), 2, (255, 255, 0), 3)      # cyan
                    cv2.putText(vis, f"C:{circularity:.2f}", (cx + 30, cy + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis, depth_text, (cx + 30, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            # ハフ変換、円
            minDist, param1, param2, minRadius, maxRadius = get_houghcircles_params('HoughCircles Control')
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
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
                        depth_m = depth_value * rs_d435i.depth_scale
                        depth_text = f"{depth_m:.3f}m"

                    # ここで カメラ3D→ロボ座標 に変換（project_center_to_robot を使用）
                    cam3d, rob3d = project_center_to_robot(
                        u=x, v=y,
                        depth_image=depth_image,
                        depth_scale=rs_d435i.depth_scale,
                        intr=rs_d435i.intr,
                        T_cam2rob=T_cam2rob,
                        roi=7  # 2m想定で少し広めに中央値を取る
                    )
                    if cam3d is not None:
                        Xc, Yc, Zc = cam3d
                        Xr, Yr, Zr = rob3d
                        # 読みやすいように別色で座標を表示
                        cv2.putText(vis, f"Cam[{Xc:.3f},{Yc:.3f},{Zc:.3f}]m",
                                    (x - 100, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis, f"Rob[{Xr:.3f},{Yr:.3f},{Zr:.3f}]m",
                                    (x - 100, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    else:
                        depth_text = "Depth: N/A"
                    # 中心座標と半径・depthを表示
                    cv2.putText(vis, depth_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 各画像を表示
            cv2.imshow('Input', overlay)
            cv2.imshow('Depth Filter', filtered_image)
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('HSV Mask', hsv_mask)
            cv2.imshow('HSV Mask Morph', mask_morph_copy)
            cv2.imshow('Result', vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('s'):
                save_params_dis(PARAM_PATH_DIS, dist_min_cm, dist_max_cm)
                save_params_hsv(PARAM_PATH_HSV[hsv_param_num], lo, hi)
                save_params_hough(PARAM_HOUGH, minDist, param1, param2, minRadius, maxRadius)
                save_noise_params(PARAM_FILTER, ftype, k, sigmaX)

    finally:
        if ARDUINO and ser is not None:
            ser.close()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # 0:赤, 1:黄, 2:青

