# -*- coding: utf-8 -*-
# 認識バージョン１
# パラメータ関係はd435iベースにやっている
# 2025/11/03 d405にも対応

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
    PARAM_FILTER,
    PARAM_HOUGH_D405,
    PARAM_PATH_DIS_D405,
    PARAM_PATH_DIS_D435I,
    PARAM_PATH_HSV,
    PARAM_HOUGH_D435I,
    CameraParam,
    compute_angles_from_position,
    encode_angle,
    encode_distance,
    load_filter_params_from_json,
    project_center_to_robot,
)

# --- 円形度ベースの円検出（Contours + Circularity） ---
# 円形度 C = 4πA / P^2 （A: 面積, P: 周長）
# 目安: 完全な円で 1.0、楕円/いびつ形で低下。0.80〜0.90 くらいが実用。
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MAX = 20000  # 大きすぎる塊を除外（必要に応じ調整）

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
    if len(args) < 3:
        print("============ Error =============================================")
        print(
            "Usage: python recognition_ver1.py [0:red, 1:yellow, 2:blue, 3:flag, 4:green, 5:teaground, 6:laf, 7:banker, 8:white, 9:blue_d405]"
        )
        print("Example for D405 camera: python recognition_ver1.py 0 d405")
        print("Example for D435i camera: python recognition_ver1.py 0 d435i")
        print("================================================================")
        return
    hsv_param_num = int(args[1])
    if args[2]=="d405":
        print("D405 mode")
        PARAM_PATH_DIS = PARAM_PATH_DIS_D405
        PARAM_HOUGH = PARAM_HOUGH_D405
    elif args[2]=="d435i":
        print("D435i mode")
        PARAM_PATH_DIS = PARAM_PATH_DIS_D435I
        PARAM_HOUGH = PARAM_HOUGH_D435I 

    # ガウシアンフィルター
    gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    # カメラクラス
    rs_d435i = CameraParam()

    pipeline = rs.pipeline()
    cfg = rs.config()

    # 解像度とFPS
    W, H, FPS = 640, 480, 15

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
    # cv2.namedWindow("Depth Filter", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Distance Control", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Control", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HoughCircles Control", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Filter GUI", cv2.WINDOW_NORMAL)

    # トラックバーを作成
    create_distance_trackbars("Distance Control")
    load_params_if_exist("Distance Control", PARAM_PATH_DIS)
    create_hsv_trackbars("HSV Control")
    load_params_if_exist("HSV Control", PARAM_PATH_HSV[hsv_param_num])
    create_houghcircles_trackbars("HoughCircles Control")
    load_params_if_exist("HoughCircles Control", PARAM_HOUGH)
    # create_noise_trackbars("Filter GUI")
    # load_params_if_exist("Filter GUI", PARAM_FILTER)

    print("[Operation]: s->Save params, q/ESC->Exit")

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
            circles = None  # ←これを追加
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
            dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m * 1000:.0f} [mm])"

            # オリジナルに載せない
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

            # グレースケール変換
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

            # ラベリング処理
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)
            mask_morph_copy = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)

            # --- ここで「円形度で良さそうな領域だけ集めるためのマスク」を用意 ---
            candidate_mask = np.zeros_like(mask_morph)  # ここに有望な領域だけ塗る
            for i in range(1, retval):  # 0は背景なのでスキップ
                x, y, w, h, area = stats[i]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])

                # 面積フィルタ（元のまま）
                if area < AREA_MIN:
                    continue

                # このラベルだけ取り出すマスクを作る
                blob_mask = np.zeros_like(mask_morph)
                blob_mask[labels == i] = 255

                # このラベル領域内だけで輪郭をとる
                roi = blob_mask[y : y + h, x : x + w]
                contours, _ = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                is_round_enough = False  # フラグ
                for cnt in contours:
                    area_cnt = cv2.contourArea(cnt)
                    if area_cnt <= 0:
                        continue

                    peri = cv2.arcLength(cnt, True)
                    if peri <= 0:
                        continue

                    circularity = (4.0 * np.pi * area_cnt) / (peri * peri)

                    # 円形度チェック
                    if circularity >= CIRC_MIN:
                        is_round_enough = True

                        # デバッグ用の可視化（今の表示は維持しつつ円形度も出す）
                        (cx_f, cy_f), r_f = cv2.minEnclosingCircle(cnt)
                        cx_abs = int(cx_f) + x
                        cy_abs = int(cy_f) + y
                        r_px = int(r_f)

                        cv2.rectangle(
                            mask_morph_copy, (x, y), (x + w, y + h), (255, 0, 0), 2
                        )
                        cv2.circle(mask_morph_copy, (cx, cy), 3, (0, 255, 255), -1)
                        cv2.putText(
                            mask_morph_copy,
                            f"[{i}]:{area} C:{circularity:.2f}",
                            (cx + 25, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3,
                        )

                        # 可視化（マゼンタ円）→元のif Falseブロックでやってたのに近い
                        cv2.circle(vis, (cx_abs, cy_abs), r_px, (255, 0, 255), 2)
                        cv2.circle(vis, (cx_abs, cy_abs), 2, (255, 255, 0), 3)
                        cv2.putText(
                            vis,
                            f"C:{circularity:.2f}",
                            (cx_abs + 30, cy_abs + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        # 輪郭が1個でも十分丸いなら、そのラベルを候補にする
                        break

                # 丸いと判断できたラベル領域だけ candidate_mask に追加
                if is_round_enough:
                    candidate_mask[labels == i] = 255

            # ハフ変換、円
            minDist, param1, param2, minRadius, maxRadius = get_houghcircles_params(
                "HoughCircles Control"
            )

            # 丸いと判断された領域だけ残した画像を作る
            if np.count_nonzero(candidate_mask) > 0:
                gray_for_hough = cv2.bitwise_and(gray, gray, mask=candidate_mask)

                circles = cv2.HoughCircles(
                    gray_for_hough,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=minDist,
                    param1=param1,
                    param2=param2,
                    minRadius=minRadius,
                    maxRadius=maxRadius,
                )

            state = "LOST"  # 可視化用（任意）
            sent = False  # このフレームで送信済みか

            if circles is not None and len(circles[0]) > 0:
                # ---- 1個だけ扱う（最初の円）----
                i = np.uint16(np.around(circles))[0][0]
                x, y, r = int(i[0]), int(i[1]), int(i[2])

                # 可視化（任意）
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)  # 外周(緑)
                cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)  # 中心(赤)

                # 深度取得
                depth_text = "Depth: N/A"
                if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                    depth_value = depth_image[y, x]
                    depth_m = depth_value * rs_d435i.depth_scale
                    depth_text = f"{depth_m:.3f}m"

                # カメラ3D→ロボ座標へ
                cam3d, rob3d = project_center_to_robot(
                    u=x,
                    v=y,
                    depth_image=depth_image,
                    depth_scale=rs_d435i.depth_scale,
                    intr=rs_d435i.intr,
                    T_cam2rob=rs_d435i.T_cam2rob,
                    roi=7,
                )

                if cam3d is not None:
                    # ---- TRACK: 検出あり ----
                    Xc, Yc, Zc = cam3d
                    Xr, Yr, Zr = rob3d

                    # 距離[mm]（ロボ座標: X-Y 平面）
                    dist_rob = round(np.sqrt(Xr**2 + Yr**2) * 1000)

                    # 角度[deg]（ロボ進行方向=+Y基準）
                    angle_deg_raw = round(compute_angles_from_position(Xr, Yr))

                    # === EMA平滑化 ===
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
                        (x - 100, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        f"Rob[{Xr:.3f},{Yr:.3f},{Zr:.3f}]m",
                        (x - 100, y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        f"D_rob:{dist_mm}mm",
                        (x - 100, y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        f"Angle:{angle_deg}deg",
                        (x - 100, y + 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        depth_text,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

                    # 送信（TRACK: '1'）
                    if ARDUINO:
                        angle_code = encode_angle(angle_deg)
                        dist_code = encode_distance(1, dist_mm)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                            # print(f"Sent(TRACK): {msg.strip()}")
                        except Exception as e:
                            print("Failed to write to serial:", e)
                    sent = True
                else:
                    # cam3d 取得失敗 → 見失い扱いにフォールバック
                    pass

            # ---- 検出なし or cam3d取得失敗 → HOLD / LOST ----
            if not sent:
                miss_count += 1
                if miss_count <= MISS_LIMIT:
                    # HOLD: 直前値を維持して送信（'2'）
                    state = f"HOLD {miss_count}/{MISS_LIMIT}"
                    if ARDUINO:
                        angle_code = encode_angle(prev_angle)
                        dist_code = encode_distance(1, prev_dist)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                            # print(f"Sent(HOLD): {msg.strip()}")
                        except Exception as e:
                            print("Failed to write to serial:", e)
                else:
                    # LOST: 安全化（ゼロ送信、直前値もリセット）（'0'）
                    state = "LOST"
                    prev_angle = 0
                    prev_dist = 0
                    if ARDUINO:
                        msg = "0000000000\n"  # mode='0', angle='0000', dist='0000'
                        try:
                            ser.write(msg.encode("ascii"))
                            # print(f"Sent(LOST): {msg.strip()}")
                        except Exception as e:
                            print("Failed to write to serial:", e)

            # 画面左上に状態を表示（任意）
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
                # cv2.imshow("Depth Filter", filtered_image)
                # cv2.imshow("Depth", depth_colormap)
                # cv2.imshow("HSV Mask", hsv_mask)
                cv2.imshow("HSV Mask Morph", mask_morph_copy)
                cv2.imshow("Result", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("s"):
                save_params_hsv(PARAM_PATH_HSV[hsv_param_num], lo, hi)
                save_params_hough(
                    PARAM_HOUGH_D405, minDist, param1, param2, minRadius, maxRadius
                )
                save_params_dis(PARAM_PATH_DIS_D405, dist_min_cm, dist_max_cm)

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


if __name__ == "__main__":
    main()  # 0:赤, 1:黄, 2:青
