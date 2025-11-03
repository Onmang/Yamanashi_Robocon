# -*- coding: utf-8 -*-
# グリーンから経路を探す

# -*- coding: utf-8 -*-
# 実装 ver1
# 各パラメータはjsonファイルで管理
# 2025/11/03 カメラ変更機能追加

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
    compute_center_distance,
    change_camera,
    get_rgbd_images,
    PARAM_PATH_DIS_D435I,
    PARAM_PATH_HSV,
    PARAM_PATH_DIS_D405,
    PARAM_HOUGH_D405,
    PARAM_PATH_DIS_GREEN,
    PARAM_HOUGH_D435I,
    PARAM_FILTER
)

# debug
DEBUG = True  # True: デバッグモードON, False: デバッグモードOFF
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
# AREA_MAX = 10000  # 大きすぎる塊を除外（必要に応じ調整）
CHANGE_CAMERA_THRE_D435I = 350 # mm
CHANGE_CAMERA_THRE_D405 = 550 # mm

# arduino シリアル通信設定
ARDUINO = False 
if ARDUINO:
    global ser

    serial_port = "/dev/ttyACM0"  # arduino UNO
    baud_rate = 115200  # 9600, 115200
    ser = serial.Serial(
        serial_port,
        baud_rate,  # できれば 115200 を推奨
        timeout=1,  # 読み取りは非ブロッキング（読みはしてないが安全）
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
    flag_lo, flag_hi = load_hsv_from_json(PARAM_PATH_HSV[3])
    green_lo, green_hi = load_hsv_from_json(PARAM_PATH_HSV[4])
    teag_lo, teag_hi = load_hsv_from_json(PARAM_PATH_HSV[5])
    laf_lo, laf_hi = load_hsv_from_json(PARAM_PATH_HSV[6])
    banker_lo, banker_hi = load_hsv_from_json(PARAM_PATH_HSV[7])
    white_lo, white_hi = load_hsv_from_json(PARAM_PATH_HSV[8])

    # ガウシアンフィルター
    gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    if DEBUG:
        print(f"Flag HSV lo:{flag_lo}, hi:{flag_hi}")
        print(f"Green HSV lo:{green_lo}, hi:{green_hi}")
        print(f"Teag HSV lo:{teag_lo}, hi:{teag_hi}")
        print(f"LAF HSV lo:{laf_lo}, hi:{laf_hi}")
        print(f"Banker HSV lo:{banker_lo}, hi:{banker_hi}")
        print(f"Gaussian Filter: k={gaus_k}, sigmaX={sigmaX}")
        print(f"White HSV lo:{white_lo}, hi:{white_hi}")

    # --------------------------------------------
    # カメラ初期化
    # --------------------------------------------
    # 解像度とFPS
    W, H, FPS = 640, 480, 15

    # RealSense D435i カメラ初期化
    cam_d435i = init_realsense_camera(
        name="d435i",
        serial="949122070535",  # 実機のシリアル
        width=W,
        height=H,
        fps=FPS,
        extrinsic_guess={
            "tx": -(32.5 * 0.001),
            "ty": -50 * 0.001,
            "tz": 200 * 0.001,
            "rx_deg": -103,
            "ry_deg": 0,
            "rz_deg": 0,
        },
        dis_param_path=PARAM_PATH_DIS_GREEN,
        hough_param_path=PARAM_HOUGH_D435I,
        ball_hsv_param_path=PARAM_PATH_HSV[2],
    )

    # --------------------------------------------
    # windown関係
    # --------------------------------------------
    if DEBUG:
        # 作成
        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Gaussian Filter", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        # サイズ変更
        w_re = 450
        h_re = 350
        cv2.resizeWindow("Input", w_re, h_re)
        # cv2.resizeWindow("Gaussian Filter", w_re, h_re)
        # cv2.resizeWindow("HSV Mask", w_re, h_re)
        cv2.resizeWindow("HSV Mask Morph", w_re, h_re)
        cv2.resizeWindow("Result", w_re, h_re)

        # 移動
        # offset_x = 50
        # offset_y = 50
        # cv2.moveWindow("Input", offset_x, offset_y)
        # cv2.moveWindow("Gaussian Filter", w_re + offset_x, offset_y)
        # cv2.moveWindow("HSV Mask", offset_x, h_re + offset_y)
        # cv2.moveWindow("HSV Mask Morph", w_re + offset_x, h_re + offset_y)
        # cv2.moveWindow("Result", 2 * w_re + offset_x, h_re + offset_y)

    # --------------------------------------------
    # メインループ
    # --------------------------------------------
    try:
        # アクティブカメラ
        activate_cam = cam_d435i
        mode = 1
        # 送信フラグ
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 5  # 短期見失いの許容量（フレーム数）

        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ

        print("main loop....")
        # main loop
        while True:
            circles = None

            # get rgbd images
            color_image, depth_image = get_rgbd_images(activate_cam)

            # センター距離計算
            if DEBUG:
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
                dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_mm:.0f} [mm])"

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

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, activate_cam.dist_min_raw, activate_cam.dist_max_raw)

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            ## ガウシアンフィルター ##
            filtered_image = cv2.GaussianBlur(filtered_image, (gaus_k, gaus_k), sigmaX)

            ## HSVマスク作成 ##
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv, np.array(white_lo, np.uint8), np.array(white_hi, np.uint8)
            )

            ## モルフォロジー変換（オープニング＋クロージング）##
            kernel = np.ones((3, 3), np.uint8)
            mask_morph = cv2.morphologyEx(
                hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3
            )
            mask_morph = cv2.morphologyEx(
                mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3
            )

            # モルフォロジーマスク適用
            vis = cv2.bitwise_and(
                filtered_image, filtered_image, mask=mask_morph
            ).copy()

            ## ラベリング処理 ##
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_morph)
            # mask_morph は 0/255 の2値
            cx_l, cy_l, area_l, bbox = largest_component_centroid(
                retval, labels, stats, centroids, area_min=AREA_MIN
            )

            # 送信準備
            state = "LOST"  # 可視化用
            sent = False  # このフレームで送信済みか

            if cx_l is not None:
                # 可視化：BBoxと重心
                x, y, w, h = bbox
                u, v = int(round(cx_l)), int(round(cy_l))
                # --- vis 側に描画 ---
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    vis,
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
                    depth_m = depth_value * activate_cam.depth_scale
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
                    depth_scale=activate_cam.depth_scale,
                    intr=activate_cam.intr,
                    T_cam2rob=activate_cam.T_cam2rob,
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

            # ---- 検出なし or cam3d取得失敗 → HOLD / LOST ----
            if not sent:
                miss_count += 1
                if miss_count <= MISS_LIMIT:
                    # HOLD: 直前値を維持して送信
                    state = f"HOLD {miss_count}/{MISS_LIMIT}"
                    if ARDUINO:
                        angle_code = encode_angle(prev_angle)
                        dist_code = encode_distance(1, prev_dist)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                            print(f"Sent(HOLD): {msg.strip()}")
                        except Exception:
                            pass
                            # print("Failed to write to serial:", e)
                else:
                    # LOST: 安全化（ゼロ送信、直前値もリセット）
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
                # cv2.imshow("Gaussian Filter", filtered_image)
                # cv2.imshow("HSV Mask", hsv_mask)
                cv2.imshow("HSV Mask Morph", mask_morph)
                cv2.imshow("Result", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()


def largest_component_centroid(retval, labels, stats, centroids, area_min=100):
    """
    2値画像(0/255)のラベリングから最大面積ラベルを選び、その重心を返す。
    bin_mask: 2値画像 (0/255)
    area_min: 面積の最小値フィルタ
    Returns:
        (cx, cy, area, bbox) or (None, None, 0, None)
    """
    if retval <= 1:
        return None, None, 0, None  # 前景なし

    # 背景(0)を除外
    areas = stats[1:, cv2.CC_STAT_AREA]
    xs = stats[1:, cv2.CC_STAT_LEFT]
    ys = stats[1:, cv2.CC_STAT_TOP]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]

    # 面積フィルタ
    valid = areas > area_min
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
    main()
