# -*- coding: utf-8 -*-
# 実装　ver1
# 各パラメータはjsonファイルで管理

import time

import cv2
import numpy as np
import serial

from common_function import (
    compute_angles_from_position,
    encode_angle,
    encode_distance,
    init_realsense_camera,
    load_filter_distance_from_json,
    load_filter_params_from_json,
    load_hough_params_from_json,
    load_hsv_from_json,
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
]  # 保存先パス選択
PARAM_HOUGH = "houghcircles_params.json"
PARAM_FILTER = "filter_params.json"  # ノイズフィルタGUIの保存先

# debug
DEBUG = True  # True: デバッグモードON, False: デバッグモードOFF

# arduino シリアル通信設定
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
    if DEBUG:
        print("Serial Port was opened:", serial_port)


def main():
    # --------------------------------------------
    # パラメータ読み込み
    # --------------------------------------------
    # hsvパラメータ読み込み
    ball_lo, ball_hi = load_hsv_from_json(PARAM_PATH_HSV[2])
    flag_lo, flag_hi = load_hsv_from_json(PARAM_PATH_HSV[3])
    green_lo, green_hi = load_hsv_from_json(PARAM_PATH_HSV[4])
    teag_lo, teag_hi = load_hsv_from_json(PARAM_PATH_HSV[5])
    laf_lo, laf_hi = load_hsv_from_json(PARAM_PATH_HSV[6])
    banker_lo, banker_hi = load_hsv_from_json(PARAM_PATH_HSV[7])
    if DEBUG:
        print(f"Ball HSV lo:{ball_lo}, hi:{ball_hi}")
        print(f"Flag HSV lo:{flag_lo}, hi:{flag_hi}")
        print(f"Green HSV lo:{green_lo}, hi:{green_hi}")
        print(f"Teag HSV lo:{teag_lo}, hi:{teag_hi}")
        print(f"LAF HSV lo:{laf_lo}, hi:{laf_hi}")
        print(f"Banker HSV lo:{banker_lo}, hi:{banker_hi}")

    # 距離パラメータ読み込み
    dist_min_cm, dist_max_cm = load_filter_distance_from_json(PARAM_PATH_DIS)

    # ガウシアンフィルター
    k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    # ハフ変換パラメータ読み込み
    minDist, param1, param2, minRadius, maxRadius = load_hough_params_from_json(
        PARAM_HOUGH
    )

    # --------------------------------------------
    # カメラ初期化
    # --------------------------------------------
    # 解像度とFPS
    W, H, FPS = 848, 480, 30

    # RealSense D435i カメラ初期化
    cam_d435i = init_realsense_camera(
        name="d435i",
        serial="949122070535",  # 実機のシリアル
        width=W,
        height=H,
        fps=FPS,
        extrinsic_guess={
            "tx": -(32.5 * 0.001),
            "ty": 0.0,
            "tz": 110 * 0.001,
            "rx_deg": -90,
            "ry_deg": 0,
            "rz_deg": 0,
        },
    )

    # RealSense D405 カメラ初期化
    if False:
        cam_d405 = init_realsense_camera(
            name="d405",
            serial="218622274519",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": 0.0,
                "ty": 0.0,
                "tz": 100 * 0.001,
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
        )

    # --------------------------------------------
    # windown関係
    # --------------------------------------------
    if DEBUG:
        # 作成
        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Filter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        # サイズ変更
        w_re = 640
        h_re = 360
        cv2.resizeWindow("Input", w_re, h_re)
        cv2.resizeWindow("Depth Filter", w_re, h_re)
        cv2.resizeWindow("HSV Mask", w_re, h_re)
        cv2.resizeWindow("HSV Mask Morph", w_re, h_re)
        cv2.resizeWindow("Result", w_re, h_re)

        # 移動
        offset_x = 50
        offset_y = 50
        cv2.moveWindow("Input", offset_x, offset_y)
        cv2.moveWindow("Depth Filter", w_re + offset_x, offset_y)
        cv2.moveWindow("HSV Mask", offset_x, h_re + offset_y)
        cv2.moveWindow("HSV Mask Morph", w_re + offset_x, h_re + offset_y)
        cv2.moveWindow("Result", 2 * w_re + offset_x, h_re + offset_y)

    # --------------------------------------------
    # メインループ
    # --------------------------------------------
    try:
        mode = 1
        # 送信フラグ
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 5  # 短期見失いの許容量（フレーム数）

        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ

        # main loop
        while True:
            # Get frameset of color and depth
            frames = cam_d435i.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = cam_d435i.align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # センター距離計算
            if DEBUG:
                # 中心座標
                cx, cy = W // 2, H // 2
                center_dist_m, center_dist_mm, _ = compute_center_distance(
                    depth_image,
                    cam_d435i.depth_scale,  # ← d435iでもd405でもOK
                    W,
                    H,
                    cx,
                    cy,
                    roi_size=10,
                )
                dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m * 1000:.0f} [mm])"

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

            # 距離によるフィルタリング
            dist_min_raw = (dist_min_cm / 100.0) / cam_d435i.depth_scale
            dist_max_raw = (dist_max_cm / 100.0) / cam_d435i.depth_scale

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, int(dist_min_raw), int(dist_max_raw))

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            # ガウシアンフィルター
            filtered_image = cv2.GaussianBlur(
                filtered_image, (k, k), sigmaX if sigmaX > 0 else 0
            )

            # HSVマスク作成
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv, np.array(ball_lo, np.uint8), np.array(ball_hi, np.uint8)
            )

            # モルフォロジー変換（オープニング＋クロージング）
            kernel = np.ones((3, 3), np.uint8)
            mask_morph = cv2.morphologyEx(
                hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3
            )
            mask_morph = cv2.morphologyEx(
                mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3
            )

            # モルフォロジーマスク適用
            vis = cv2.bitwise_and(filtered_image, filtered_image, mask=mask_morph)

            # グレースケール変換
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

            # ハフ変換で円検出
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius,
            )

            # 送信準備
            state = "LOST"  # 可視化用（任意）
            sent = False  # このフレームで送信済みか

            # 一番最初の円が最も円らしい
            if circles is not None and len(circles[0]) > 0:
                # ---- 1個だけ扱う（最初の円）----
                i = np.uint16(np.around(circles))[0][0]
                x, y, _ = int(i[0]), int(i[1]), int(i[2])

                # カメラ3D→ロボ座標へ
                cam3d, rob3d = project_center_to_robot(
                    u=x,
                    v=y,
                    depth_image=depth_image,
                    depth_scale=cam_d435i.depth_scale,
                    intr=cam_d435i.intr,
                    T_cam2rob=cam_d435i.T_cam2rob,
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
                    if DEBUG:
                        # 左上に固定して表示
                        tx, ty = 30, 50  # 表示開始位置（左上からのオフセット）
                        line_h = 40  # 行間ピクセル
                        cv2.putText(
                            vis,
                            f"D_rob:{dist_mm}mm",
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            vis,
                            f"Angle:{angle_deg}deg",
                            (tx, ty + line_h),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
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
            if DEBUG:
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
            if DEBUG:
                cv2.imshow("Input", overlay)
                cv2.imshow("Depth Filter", filtered_image)
                cv2.imshow("HSV Mask", hsv_mask)
                cv2.imshow("HSV Mask Morph", mask_morph)
                cv2.imshow("Result", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        # cam_d405.pipeline.stop()
        cv2.destroyAllWindows()


def compute_center_distance(depth_image, depth_scale, W, H, cx, cy, roi_size=10):
    """
    画像中心付近(roi_size x roi_size)の深度の平均値から距離を求める関数

    Args:
        depth_image (ndarray): 深度画像 (uint16など、RealSenseのZ16想定)
        depth_scale (float): RealSenseのdepth_scale [m/1depth_unit]
        W (int): 画像の幅
        H (int): 画像の高さ
        roi_size (int): 中心から取る正方形ROIの一辺ピクセル数

    Returns:
        center_dist_m (float): 中心近傍の平均距離 [m]
        center_dist_mm (float): 中心近傍の平均距離 [mm]
        avg_dist_raw (float): 深度の生値平均 (スケールかける前, depth単位)
    """

    # ROIの範囲（切り出しの安全ガード付き）
    half = roi_size // 2
    x1, x2 = max(0, cx - half), min(W, cx + half)
    y1, y2 = max(0, cy - half), min(H, cy + half)

    # 中心領域の切り出し
    center_roi = depth_image[y1:y2, x1:x2]

    # 深度0(=無効)を除いた平均
    non_zero_values = center_roi[center_roi > 0]
    if non_zero_values.size > 0:
        avg_dist_raw = float(np.mean(non_zero_values))
    else:
        avg_dist_raw = 0.0

    # スケール適用
    center_dist_m = avg_dist_raw * depth_scale
    center_dist_mm = center_dist_m * 1000.0

    return center_dist_m, center_dist_mm, avg_dist_raw


if __name__ == "__main__":
    main()
