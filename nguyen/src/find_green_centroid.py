# -*- coding: utf-8 -*-
# 認識バージョン１
# パラメータ関係はd435iベースにやっている

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import serial

from common_function import (
    CameraParam,
    compute_angles_from_position,
    encode_angle,
    encode_distance,
    load_filter_params_from_json,
    project_center_to_robot,
    
)

# パラメータ保存用のファイルパス
PARAM_PATH_DIS = "distance_params.json"
PARAM_PATH_HSV = [
    "hsv_params_red.json",
    "hsv_params_yellow.json",
    "hsv_params_blue.json",
    "hsv_params_flag.json",
    "hsv_params_green.json",
    "hsv_params_teaground.json",
    "hsv_params_laf.json",
    "hsv_params_banker.json",
    "hsv_params_white.json",
]  # 保存先パス選択
PARAM_HOUGH = "houghcircles_params.json"
PARAM_FILTER = "gaussian_filter_params.json"  # ノイズフィルタGUIの保存先


# --- 円形度ベースの円検出（Contours + Circularity） ---
# 円形度 C = 4πA / P^2 （A: 面積, P: 周長）
# 目安: 完全な円で 1.0、楕円/いびつ形で低下。0.80〜0.90 くらいが実用。
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MAX = 10000  # 大きすぎる塊を除外（必要に応じ調整）

# Arduino接続設定
ARDUINO = False
if ARDUINO:
    global ser

    serial_port = "/dev/ttyACM0"  # arduino UNO
    # serial_port = '/dev/ttyACM0'
    # takemichi arduino nano evry
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
    print("Serial Port was opened:", serial_port)


def main():
    # argv
    args = sys.argv
    if len(args) < 2:
        print("============ Error =============================================")
        print(
            "Usage: python recognition_ver1.py [0:red, 1:yellow, 2:blue, 3:flag, 4:green, 5:teaground, 6:laf, 7:banker, 8:white]"
        )
        print("================================================================")
        return
    hsv_param_num = int(args[1])

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
    intr = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    # クラス格納
    rs_d435i.add_ins_param(intr)
    inst_matrix = np.array(
        [[rs_d435i.fx, 0, rs_d435i.ppx], [0, rs_d435i.fy, rs_d435i.ppy], [0, 0, 1]]
    )
    print(f"Camera Intrinsics: {rs_d435i.intr.width}x{rs_d435i.intr.height}")
    print("Intrinsic Matrix:")
    print("[[fx, 0, ppx],")
    print(" [0, fy, ppy],")
    print(" [0, 0, 1]]")
    print(inst_matrix)

    # ガウシアンフィルター
    gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    # 深度センサーの情報を取得
    depth_sensor = profile.get_device().first_depth_sensor()
    rs_d435i.set_transform(depth_sensor.get_option(rs.option.stereo_baseline))
    print(f"Stereo Baseline is: {rs_d435i.stereo_baseline} mm")

    # 深度センサーからdepth_scaleを取得（単位をメートルに変換する係数）
    rs_d435i.depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {rs_d435i.depth_scale}")
    print(f"Camera to Robot Transform:\n{rs_d435i.T_cam2rob}")
    # return

    # 深度とカラーの位置合わせ（Align）オブジェクトを作成
    align_to = rs.stream.color
    align = rs.align(align_to)

    # ウィンドウの作成と配置
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Filter", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Distance Control", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Control", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # トラックバーを作成
    create_distance_trackbars("Distance Control")
    load_params_if_exist("Distance Control", PARAM_PATH_DIS)
    create_hsv_trackbars("HSV Control")
    load_params_if_exist("HSV Control", PARAM_PATH_HSV[hsv_param_num])

    # ウィンドウが重ならないように初期位置を設定
    # win_w, win_h = 450, 350  # ウィンドウサイズを小さく調整
    # offset_x = 50
    # offset_y = 50
    # cv2.moveWindow("Input", offset_x, offset_y)
    # cv2.moveWindow("Depth Filter", win_w + offset_x, offset_y)
    # cv2.moveWindow("Depth", 2 * win_w + offset_x, offset_y)
    # cv2.moveWindow("HSV Mask", offset_x, win_h + offset_y)
    # cv2.moveWindow("HSV Mask Morph", win_w + offset_x, win_h + offset_y)
    # cv2.moveWindow("Result", 2 * win_w + offset_x, win_h + offset_y)

    # cv2.moveWindow("Distance Control", 3 * win_w + offset_x, offset_y)
    # cv2.moveWindow("HSV Control", 3 * win_w + offset_x, win_h + offset_y)
    # print("[Operation]: s->Save params, q/ESC->Exit")

    # arduino　送信設定
    mode = 1
    angle_deg = 0

    try:
        # 送信フラグ
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 5  # 短期見失いの許容量（フレーム数）

        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ

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
            roi_size = 20
            half = roi_size // 2
            x1 = int(cx - half)
            y1 = int(cy - half)
            x2 = int(cx + half - 1)
            y2 = int(cy + half - 1)

            # 中心領域
            center_roi = depth_image[y1:y2, x1:x2]

            # 0を除く平均値を計算
            non_zero_values = center_roi[center_roi > 0]
            avg_dist_raw = np.mean(non_zero_values) if non_zero_values.size > 0 else 0

            # 正しいdepth_scaleを使ってメートル[m]に変換
            center_dist_m = avg_dist_raw * rs_d435i.depth_scale
            dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m * 1000:.0f} [mm])"

            # オリジナルに載せない
            overlay = color_image.copy()

            # 中心領域を矩形で表示
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color=(255, 255, 255),  # 白枠
                thickness=1,
            )

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

            # トラックバーから距離の範囲を[cm]で取得
            dist_min_cm, dist_max_cm = get_distance_range("Distance Control")

            # dist_min が dist_max を超えた場合、値を入れ替える
            if dist_min_cm > dist_max_cm:
                dist_min_cm, dist_max_cm = dist_max_cm, dist_min_cm
                cv2.setTrackbarPos("Dist_min [cm]", "Distance Control", dist_min_cm)
                cv2.setTrackbarPos("Dist_max [cm]", "Distance Control", dist_max_cm)

            # 距離[cm]を、正しいdepth_scaleを使ってraw深度値に変換
            dist_min_raw = (dist_min_cm / 100.0) / rs_d435i.depth_scale
            dist_max_raw = (dist_max_cm / 100.0) / rs_d435i.depth_scale

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, int(dist_min_raw), int(dist_max_raw))

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            # 深度画像を可視化用にカラーマップに変換
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            ## ガウシアンフィルター ##
            filtered_image = cv2.GaussianBlur(filtered_image, (gaus_k, gaus_k), sigmaX)

            # HSV変換して色抽出
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            lo, hi = get_hsv_range("HSV Control")
            hsv_mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

            # 膨張・収縮でマスク整形（任意）
            kernel = np.ones((3, 3), np.uint8)
            mask_morph = cv2.morphologyEx(
                hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3
            )
            mask_morph = cv2.morphologyEx(
                mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3
            )

            # 可視化（マスクをカラーに適用）
            vis = cv2.bitwise_and(
                filtered_image, filtered_image, mask=mask_morph
            ).copy()

            # --- ラベリング（HSVモルフォロジ後のマスク）---
            # mask_morph は 0/255 の2値
            cx_l, cy_l, area_l, bbox = largest_component_centroid(
                mask_morph, area_min=AREA_MIN
            )

            # ここで表示用にBGR化したマスクを用意（←描くキャンバス）
            mask_vis = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)

            state = "LOST"
            sent = False

            if cx_l is not None:
                # 可視化：BBoxと重心
                x, y, w, h = bbox
                u, v = int(round(cx_l)), int(round(cy_l))
                # --- vis 側に描画 ---
                cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.drawMarker(mask_vis, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    mask_vis,
                    f"Area:{area_l}  C:({u},{v})",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                # 深度表示（任意）
                depth_text = "Depth: N/A"
                if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                    depth_value = depth_image[v, u]
                    depth_m = depth_value * rs_d435i.depth_scale
                    depth_text = f"{depth_m:.3f}m"
                cv2.putText(
                    vis,
                    depth_text,
                    (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                # 3D投影 → ロボ座標
                cam3d, rob3d = project_center_to_robot(
                    u=u,
                    v=v,
                    depth_image=depth_image,
                    depth_scale=rs_d435i.depth_scale,
                    intr=rs_d435i.intr,
                    T_cam2rob=rs_d435i.T_cam2rob,
                    roi=7,
                )

                if cam3d is not None:
                    Xc, Yc, Zc = cam3d
                    Xr, Yr, Zr = rob3d

                    # 距離・角度
                    dist_rob = round(np.sqrt(Xr**2 + Yr**2) * 1000)  # [mm]
                    angle_deg_raw = round(compute_angles_from_position(Xr, Yr))

                    # 平滑化
                    angle_deg = round(
                        EMA_ALPHA * angle_deg_raw + (1 - EMA_ALPHA) * prev_angle
                    )
                    dist_mm = round(EMA_ALPHA * dist_rob + (1 - EMA_ALPHA) * prev_dist)

                    # 更新
                    prev_angle = angle_deg
                    prev_dist = dist_mm
                    miss_count = 0
                    state = "TRACK"

                    # 表示
                    cv2.putText(
                        vis,
                        f"Cam[{Xc:.3f},{Yc:.3f},{Zc:.3f}]m",
                        (u - 100, v + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        vis,
                        f"Rob[{Xr:.3f},{Yr:.3f},{Zr:.3f}]m",
                        (u - 100, v + 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        vis,
                        f"D_rob:{dist_mm}mm",
                        (u - 100, v + 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        vis,
                        f"Angle:{angle_deg}deg",
                        (u - 100, v + 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    # 送信
                    if ARDUINO:
                        angle_code = encode_angle(angle_deg)
                        dist_code = encode_distance(1, dist_mm)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                        except Exception as e:
                            print("Failed to write to serial:", e)
                    sent = True

            # 検出なし（重心無し）→ HOLD/LOST の既存処理をそのまま下に残す
            if not sent:
                miss_count += 1
                if miss_count <= MISS_LIMIT:
                    state = f"HOLD {miss_count}/{MISS_LIMIT}"
                    if ARDUINO:
                        angle_code = encode_angle(prev_angle)
                        dist_code = encode_distance(1, prev_dist)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                        except Exception as e:
                            print("Failed to write to serial:", e)
                else:
                    state = "LOST"
                    prev_angle = 0
                    prev_dist = 0
                    if ARDUINO:
                        msg = "0000000000\n"
                        try:
                            ser.write(msg.encode("ascii"))
                        except Exception as e:
                            print("Failed to write to serial:", e)

            # 状態表示（既存のまま）
            cv2.putText(
                vis,
                f"STATE: {state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 200, 255)
                if "HOLD" in state
                else ((0, 255, 0) if state == "TRACK" else (0, 0, 255)),
                2,
            )

            # 各画像を表示
            if True:
                cv2.imshow("Input", overlay)
                cv2.imshow("Depth Filter", filtered_image)
                cv2.imshow("Depth", depth_colormap)
                cv2.imshow("HSV Mask", hsv_mask)
                cv2.imshow("HSV Mask Morph", mask_vis)
                cv2.imshow("Result", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("s"):
                save_params_dis(PARAM_PATH_DIS, dist_min_cm, dist_max_cm)
                save_params_hsv(PARAM_PATH_HSV[hsv_param_num], lo, hi)

    finally:
        if ARDUINO and ser is not None:
            ser.close()
        pipeline.stop()
        cv2.destroyAllWindows()


def _noop(x):
    pass


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
    data = {"filter_type": ftype, "ksize": k, "sigmaX": sigmaX}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved Filter params -> {path}")


def create_houghcircles_trackbars(win):
    cv2.createTrackbar("minDist", win, 50, 500, _noop)
    cv2.createTrackbar("param1", win, 100, 300, _noop)
    cv2.createTrackbar("param2", win, 30, 150, _noop)
    cv2.createTrackbar("minRadius", win, 10, 100, _noop)
    cv2.createTrackbar("maxRadius", win, 100, 200, _noop)


def get_houghcircles_params(win):
    minDist = cv2.getTrackbarPos("minDist", win)
    param1 = cv2.getTrackbarPos("param1", win)
    param2 = cv2.getTrackbarPos("param2", win)
    minRadius = cv2.getTrackbarPos("minRadius", win)
    maxRadius = cv2.getTrackbarPos("maxRadius", win)
    return minDist, param1, param2, minRadius, maxRadius


def save_params_hough(path, minDist, param1, param2, minRadius, maxRadius):
    """パラメータをJSONファイルに保存する"""
    data = {
        "minDist": minDist,
        "param1": param1,
        "param2": param2,
        "minRadius": minRadius,
        "maxRadius": maxRadius,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved HoughCircles params -> {path}")


def create_hsv_trackbars(win):
    cv2.createTrackbar("H_low", win, 0, 179, _noop)
    cv2.createTrackbar("H_high", win, 179, 179, _noop)
    cv2.createTrackbar("S_low", win, 0, 255, _noop)
    cv2.createTrackbar("S_high", win, 255, 255, _noop)
    cv2.createTrackbar("V_low", win, 0, 255, _noop)
    cv2.createTrackbar("V_high", win, 255, 255, _noop)


def get_hsv_range(win):
    hl = cv2.getTrackbarPos("H_low", win)
    hh = cv2.getTrackbarPos("H_high", win)
    sl = cv2.getTrackbarPos("S_low", win)
    sh = cv2.getTrackbarPos("S_high", win)
    vl = cv2.getTrackbarPos("V_low", win)
    vh = cv2.getTrackbarPos("V_high", win)
    return (hl, sl, vl), (hh, sh, vh)


def create_distance_trackbars(win, max_dist_cm=400):
    """距離[cm]を調整するトラックバーを作成する"""
    # D405を想定した初期値 (7cm - 50cm)
    cv2.createTrackbar("Dist_min [cm]", win, 7, max_dist_cm, _noop)
    cv2.createTrackbar("Dist_max [cm]", win, 50, max_dist_cm, _noop)


def get_distance_range(win):
    """トラックバーから距離[cm]の範囲を取得する"""
    d_min_cm = cv2.getTrackbarPos("Dist_min [cm]", win)
    d_max_cm = cv2.getTrackbarPos("Dist_max [cm]", win)
    return d_min_cm, d_max_cm


def save_params_dis(path, d_min_cm, d_max_cm):
    """パラメータをJSONファイルに保存する"""
    data = {"Dist_min [cm]": d_min_cm, "Dist_max [cm]": d_max_cm}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved Distance params -> {path}")


def save_params_hsv(path, lo, hi):
    data = {
        "H_low": lo[0],
        "S_low": lo[1],
        "V_low": lo[2],
        "H_high": hi[0],
        "S_high": hi[1],
        "V_high": hi[2],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved HSV params -> {path}")


def load_params_if_exist(win, path):
    """パラメータが既存ならJSONファイルから読み込む"""
    p = Path(path)
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in data.items():
        cv2.setTrackbarPos(k, win, int(v))
    print(f"Loaded params <- {path}")


def show_hist(img_hsv):
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    plt.figure(figsize=(12, 8))

    plt.plot(hist_h, color="r", label="h")
    plt.plot(hist_s, color="g", label="s")
    plt.plot(hist_v, color="b", label="v")
    plt.legend()
    plt.show()


def largest_component_centroid(bin_mask, area_min=100):
    """
    2値画像(0/255)のラベリングから最大面積ラベルを選び、その重心を返す。
    bin_mask: 2値画像 (0/255)
    area_min: 面積の最小値フィルタ
    Returns:
        (cx, cy, area, bbox) or (None, None, 0, None)
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    if num <= 1:
        return None, None, 0, None  # 前景なし

    # 背景(0)を除外
    areas = stats[1:, cv2.CC_STAT_AREA]
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    # 面積フィルタ
    valid = areas >= area_min
    if not np.any(valid):
        return None, None, 0, None

    idx_rel = np.argmax(areas * valid)  # 有効範囲で最大
    if not valid[idx_rel]:
        return None, None, 0, None

    idx = idx_rel + 1  # 背景分 +1
    cx, cy = centroids[idx]
    bbox = (int(xs[idx_rel]), int(ys[idx_rel]), int(ws[idx_rel]), int(hs[idx_rel]))
    return float(cx), float(cy), int(areas[idx_rel]), bbox


if __name__ == "__main__":
    main()  # 0:赤, 1:黄, 2:青
