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
    PARAM_PATH_DIS,
    PARAM_PATH_HSV,
    PARAM_HOUGH,
    PARAM_FILTER
)

# debug
DEBUG = False  # True: デバッグモードON, False: デバッグモードOFF
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MAX = 10000  # 大きすぎる塊を除外（必要に応じ調整）
CHANGE_CAMERA_THRE_D435I = 350 # mm
CHANGE_CAMERA_THRE_D405 = 550 # mm

# arduino シリアル通信設定
ARDUINO = True
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
    ball_lo, ball_hi = load_hsv_from_json(PARAM_PATH_HSV[2])
    flag_lo, flag_hi = load_hsv_from_json(PARAM_PATH_HSV[3])
    green_lo, green_hi = load_hsv_from_json(PARAM_PATH_HSV[4])
    teag_lo, teag_hi = load_hsv_from_json(PARAM_PATH_HSV[5])
    laf_lo, laf_hi = load_hsv_from_json(PARAM_PATH_HSV[6])
    banker_lo, banker_hi = load_hsv_from_json(PARAM_PATH_HSV[7])

    # 距離パラメータ読み込み
    dist_min_cm, dist_max_cm = load_filter_distance_from_json(PARAM_PATH_DIS)

    # ガウシアンフィルター
    gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    # ハフ変換パラメータ読み込み
    minDist, param1, param2, minRadius, maxRadius = load_hough_params_from_json(
        PARAM_HOUGH
    )
    if DEBUG:
        print(f"Ball HSV lo:{ball_lo}, hi:{ball_hi}")
        print(f"Flag HSV lo:{flag_lo}, hi:{flag_hi}")
        print(f"Green HSV lo:{green_lo}, hi:{green_hi}")
        print(f"Teag HSV lo:{teag_lo}, hi:{teag_hi}")
        print(f"LAF HSV lo:{laf_lo}, hi:{laf_hi}")
        print(f"Banker HSV lo:{banker_lo}, hi:{banker_hi}")
        print(f"Gaussian Filter: k={gaus_k}, sigmaX={sigmaX}")
        print(f"Distance Filter: min={dist_min_cm}cm, max={dist_max_cm}cm")
        print(
            f"HoughCircles: minDist={minDist}, param1={param1}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}"
        )

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
    )

    # RealSense D405 カメラ初期化
    # d405はcam3d
    if True:
        cam_d405 = init_realsense_camera(
            name="d405",        
            serial="218622274519",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": 0.0,
                "ty": 0.0,
                "tz": 0.0,
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
        cv2.namedWindow("Gaussian Filter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        # サイズ変更
        w_re = 450
        h_re = 350
        cv2.resizeWindow("Input", w_re, h_re)
        cv2.resizeWindow("Gaussian Filter", w_re, h_re)
        cv2.resizeWindow("HSV Mask", w_re, h_re)
        cv2.resizeWindow("HSV Mask Morph", w_re, h_re)
        cv2.resizeWindow("Result", w_re, h_re)

        # 移動
        offset_x = 50
        offset_y = 50
        cv2.moveWindow("Input", offset_x, offset_y)
        cv2.moveWindow("Gaussian Filter", w_re + offset_x, offset_y)
        cv2.moveWindow("HSV Mask", offset_x, h_re + offset_y)
        cv2.moveWindow("HSV Mask Morph", w_re + offset_x, h_re + offset_y)
        cv2.moveWindow("Result", 2 * w_re + offset_x, h_re + offset_y)

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
                
                # 中心領域を矩形で表示
                cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        color=(0, 200, 0),  # 緑枠
                        thickness=1,
                    )

            ## 距離によるフィルタリング ##
            dist_min_raw = (dist_min_cm / 100.0) / activate_cam.depth_scale
            dist_max_raw = (dist_max_cm / 100.0) / activate_cam.depth_scale

            # 指定範囲内のマスクを作成
            mask = cv2.inRange(depth_image, int(dist_min_raw), int(dist_max_raw))

            # マスクを適用してフィルタリング
            filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            ## ガウシアンフィルター ##
            filtered_image = cv2.GaussianBlur(filtered_image, (gaus_k, gaus_k), sigmaX)

            ## HSVマスク作成 ##
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv, np.array(ball_lo, np.uint8), np.array(ball_hi, np.uint8)
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

            # グレースケール変換
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

            ## ラベリング処理 ##
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)
            mask_morph_copy = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)

            # 円形度良いものだけ抜き出す
            candidate_mask = np.zeros_like(mask_morph)  # ここに有望な領域だけ塗る

            for i in range(1, retval):  # 0は背景なのでスキップ
                x, y, w, h, area = stats[i]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])

                # 面積フィルタ（元のまま）
                if area < AREA_MIN or area > AREA_MAX:
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
                    if area_cnt <= 0:  # 一応
                        continue

                    peri = cv2.arcLength(cnt, True)
                    if peri <= 0:
                        continue

                    circularity = (4.0 * np.pi * area_cnt) / (peri * peri)

                    # 円形度チェック
                    if circularity >= CIRC_MIN:
                        is_round_enough = True

                        if DEBUG:
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

            # 送信準備
            state = "LOST"  # 可視化用
            sent = False  # このフレームで送信済みか

            # === 1. 円検出結果の評価 ===
            if circles is not None and len(circles[0]) > 0:
                # 最初の円だけ使う
                i = np.uint16(np.around(circles))[0][0]
                x, y, r = int(i[0]), int(i[1]), int(i[2])

                # 半径チェック（ノイズ除外用。調整してOK）
                if 10 <= r <= 100:
                    if DEBUG:
                        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)  # 外周(緑)
                        cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)  # 中心(赤)

                    # ピクセル→3D→ロボ座標
                    cam3d, rob3d = project_center_to_robot(
                        u=x,
                        v=y,
                        depth_image=depth_image,
                        depth_scale=activate_cam.depth_scale,
                        intr=activate_cam.intr,
                        T_cam2rob=activate_cam.T_cam2rob,
                        roi=7,
                    )

                    if cam3d is not None:
                        Xc, Yc, Zc = cam3d
                        Xr, Yr, Zr = rob3d

                        # ロボ座標での水平距離[mm]
                        dist_rob_mm = np.sqrt(Xr**2 + Yr**2) * 1000.0

                        # 距離がありえない値（極端にデカい/NaN）なら捨てる
                        if (not np.isnan(dist_rob_mm)) and (dist_rob_mm < 3000):

                            # 角度[deg] ロボ+Y基準
                            angle_deg_raw = round(compute_angles_from_position(Xr, Yr))

                            # === EMA平滑化 ===
                            angle_deg = round(
                                EMA_ALPHA * angle_deg_raw + (1 - EMA_ALPHA) * prev_angle
                            )
                            dist_mm_raw = round(dist_rob_mm)
                            dist_mm = round(
                                EMA_ALPHA * dist_mm_raw + (1 - EMA_ALPHA) * prev_dist
                            )

                            # [TEST]しきい値以内なら 0 距離を送る
                            dist_mm_thresh = 170  # mm D405の閾値
                            dist_mm_send = dist_mm if dist_mm > dist_mm_thresh else 0

                            # 前回値更新
                            prev_angle = angle_deg
                            prev_dist = dist_mm

                            # 見失いカウンタをここでだけリセット
                            miss_count = 0

                            # 表示情報
                            state = "TRACK"
                            if DEBUG:
                                tx, ty = 10, 70
                                line_h = 40
                                cv2.putText(
                                    vis,
                                    f"D_rob: {dist_mm}mm",
                                    (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                cv2.putText(
                                    vis,
                                    f"Angle: {angle_deg}deg",
                                    (tx, ty + line_h),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                            # シリアル送信
                            if ARDUINO:
                                angle_code = encode_angle(angle_deg)
                                dist_code = encode_distance(1, dist_mm_send)
                                msg = f"{mode}{angle_code}{dist_code}\n"
                                try:
                                    ser.write(msg.encode("ascii"))
                                    # print(f"Sent(TRACK): {msg.strip()}")
                                except Exception as e:
                                    print("Failed to write to serial:", e)

                            # カメラ変更判定
                            activate_cam = change_camera(activate_cam, cam_d435i, cam_d405, Zc * 1000, thre_d435i=CHANGE_CAMERA_THRE_D435I, thre_d405=CHANGE_CAMERA_THRE_D405)
                            sent = True  # 今フレームは送った

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
                cv2.imshow("Gaussian Filter", filtered_image)
                cv2.imshow("HSV Mask", hsv_mask)
                cv2.imshow("HSV Mask Morph", mask_morph_copy)
                cv2.imshow("Result", vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        cam_d405.pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
