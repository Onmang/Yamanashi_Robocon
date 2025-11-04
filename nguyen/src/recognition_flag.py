# -*- coding: utf-8 -*-
# flag 認識テスト
# 2025/11/03 とりあえず三角形検出のみ


import cv2
import numpy as np
import serial
import time

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
    PARAM_HOUGH_D435I,
    PARAM_FILTER
)

DEBUG = True  # True: デバッグモードON, False: デバッグモードOFF
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MIN_FLAG = 200 # flag用三角形最小面積

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

    # ガウシアンフィルター
    gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

    if DEBUG:
        print(f"Flag HSV lo:{flag_lo}, hi:{flag_hi}")
        print(f"Green HSV lo:{green_lo}, hi:{green_hi}")
        print(f"Teag HSV lo:{teag_lo}, hi:{teag_hi}")
        print(f"LAF HSV lo:{laf_lo}, hi:{laf_hi}")
        print(f"Banker HSV lo:{banker_lo}, hi:{banker_hi}")
        print(f"Gaussian Filter: k={gaus_k}, sigmaX={sigmaX}")

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
        dis_param_path=PARAM_PATH_DIS_D435I,
        hough_param_path=PARAM_HOUGH_D435I,
        ball_hsv_param_path=PARAM_PATH_HSV[2],
    )

    # --------------------------------------------
    # windown関係
    # --------------------------------------------
    if DEBUG:
        # 作成
        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        # サイズ変更
        w_re = 300
        h_re = 250
        cv2.resizeWindow("Input", w_re, h_re)
        cv2.resizeWindow("HSV Mask", w_re, h_re)
        cv2.resizeWindow("HSV Mask Morph", w_re, h_re)
        cv2.resizeWindow("Result", w_re, h_re)

        # # 移動
        # offset_x = 50
        # offset_y = 50
        # cv2.moveWindow("Input", offset_x, offset_y)
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
            flags = None
            
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
                dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_mm * 1000:.0f} [mm])"

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
                hsv, np.array(flag_lo, np.uint8), np.array(flag_hi, np.uint8)
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

            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)

            # 三角形検出
            triangles = detect_triangles(retval, labels, stats, area_min_label=AREA_MIN, area_min=AREA_MIN_FLAG, epsilon_ratio=0.08)

            valid_triangles = []
            valid_rob3d = []   # ← rob座標を丸ごと入れる（[Xr, Yr, Zr]）
            valid_dist_mm = [] # ← 水平距離mmをすぐ使えるように入れておく
            
            # 送信準備
            state = "LOST"  # 可視化用
            sent = False  # このフレームで送信済みか

            if triangles:
                for tri in triangles:
                    # 重心
                    M = cv2.moments(tri)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # ピクセル → カメラ/ロボ
                    cam3d, rob3d = project_center_to_robot(
                        u=cx,
                        v=cy,
                        depth_image=depth_image,
                        depth_scale=activate_cam.depth_scale,
                        intr=activate_cam.intr,
                        T_cam2rob=activate_cam.T_cam2rob,
                        roi=7,
                    )
                    if rob3d is None or np.any(np.isnan(rob3d)):
                        continue

                    Xr, Yr, Zr = rob3d  # [m]
                    # ロボ座標での水平距離[mm]
                    dist_rob_mm = np.sqrt(Xr**2 + Yr**2) * 1000.0

                    # 範囲フィルタリング
                    if 500 <= dist_rob_mm <= 3000:
                        valid_triangles.append(tri)
                        valid_rob3d.append(rob3d)
                        valid_dist_mm.append(dist_rob_mm)

                # --- 最も近い三角形を選択して、角度・距離を計算 ---
                if valid_rob3d:
                    # 一番近い水平距離を持つインデックス
                    nearest_idx = int(np.argmin(valid_dist_mm))
                    nearest_tri = valid_triangles[nearest_idx]
                    nearest_rob3d = valid_rob3d[nearest_idx]
                    
                    # 旗のポール分オフセットする
                    edge = find_vertical_edge(nearest_tri)
                    if edge is not None:
                        p1, p2 = edge
                        mx = int((p1[0] + p2[0]) / 2)
                        my = int((p1[1] + p2[1]) / 2)
                        # この1点だけを3Dにする
                        _, rob3d = project_center_to_robot(
                            u=mx,
                            v=my,
                            depth_image=depth_image,
                            depth_scale=activate_cam.depth_scale,
                            intr=activate_cam.intr,
                            T_cam2rob=activate_cam.T_cam2rob,
                            roi=7,
                        )
                        Xr, Yr, _ = rob3d
                    else:
                        Xr, Yr, _ = nearest_rob3d

                    # 角度[deg] ロボ+Y基準
                    angle_deg_raw = round(compute_angles_from_position(Xr, Yr))

                    # === EMA平滑化 ===
                    angle_deg = round(
                        EMA_ALPHA * angle_deg_raw + (1 - EMA_ALPHA) * prev_angle
                    )
                    dist_mm_raw = round(valid_dist_mm[nearest_idx])  # もうmmになってる
                    dist_mm = round(
                        EMA_ALPHA * dist_mm_raw + (1 - EMA_ALPHA) * prev_dist
                    )

                    # 前回値更新
                    prev_angle = angle_deg
                    dist_mm_send, prev_dist = 0, 0 # ゴールを探すのは距離関係ない

                    # 見失いカウンタリセット
                    miss_count = 0
                    state = "TRACK"

                    # 可視化
                    if DEBUG:
                        cv2.drawContours(vis, [nearest_tri], -1, (255, 0, 255), 3)
                        if edge is not None:
                            # 垂直辺の中点を可視化
                            cv2.line(vis, tuple(p1), tuple(p2), (0, 255, 255), 2)
                            cv2.circle(vis, (mx, my), 6, (255, 0, 255), -1)
                        else:
                            # fallback: 重心を可視化
                            M = cv2.moments(nearest_tri)
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
                        cv2.putText(
                            vis,
                            f"D_rob: {dist_mm}mm",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            vis,
                            f"Angle: {angle_deg}deg",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    # シリアル送信
                    sent = True  # 今フレームは送った
                    if ARDUINO:
                        angle_code = encode_angle(angle_deg)
                        dist_code = encode_distance(1, dist_mm_send)
                        msg = f"{mode}{angle_code}{dist_code}\n"
                        try:
                            ser.write(msg.encode("ascii"))
                        except Exception as e:
                            print("Failed to write to serial:", e)
            
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
                        msg = "0000500000\n"  # 左に5度回転続ける
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
            
            if DEBUG:
                cv2.imshow("Input", overlay)
                cv2.imshow("Result", vis)
                cv2.imshow("HSV Mask", mask)
                cv2.imshow("HSV Mask Morph", mask_morph)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()

def detect_triangles(retval, labels, stats, area_min_label=200, area_min=200, epsilon_ratio=0.08):
    """
    connectedComponentsWithStats() の結果から三角形を検出して返す関数。
    描画は外で行う。

    Args:
        retval (int): ラベル数（connectedComponentsWithStats の戻り値）
        labels (ndarray): 各ピクセルのラベル番号画像
        stats (ndarray): 各ラベルの統計情報 [x, y, w, h, area]
        vis (ndarray): 入力画像（参照のみ）
        area_min_label (int): 小さいラベルを除外する閾値
        area_min (int): 三角形の最小面積
        epsilon_ratio (float): 輪郭近似のしきい値（小さいほど形を正確に再現）

    Returns:
        approx_contours (list): 三角形の輪郭リスト（各要素はN×1×2のnumpy配列）
    """
    approx_contours = []

    for i in range(1, retval):
        x, y, w, h, area = stats[i]
        if area < area_min_label:
            continue

        blob_mask = np.uint8(labels == i) * 255
        roi = blob_mask[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            arclen = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon_ratio * arclen, True)

            if len(approx) == 3 and cv2.contourArea(approx) >= area_min:
                approx[:, 0, 0] += x
                approx[:, 0, 1] += y
                approx_contours.append(approx)
                break  # 1ラベルにつき1つでOK

    return approx_contours

def find_vertical_edge(tri, vertical_ratio=0.2):
    """
    tri: cv2.approxPolyDPで得た三角形 (3x1x2) を想定
    vertical_ratio: |dx| が |dy| の何割以下なら「縦」とみなすか
    戻り値: (p1, p2) 縦に一番近い辺の2点。見つからなければ None
    """
    pts = tri.reshape(-1, 2)  # [[x1,y1],[x2,y2],[x3,y3]]
    edges = [
        (pts[0], pts[1]),
        (pts[1], pts[2]),
        (pts[2], pts[0]),
    ]

    best_edge = None
    best_score = None  # 小さいほど縦

    for p1, p2 in edges:
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1]) + 1e-6  # 0割り防止
        score = dx / dy  # 0に近いほど縦

        if best_score is None or score < best_score:
            best_score = score
            best_edge = (p1, p2)

    # ここで「どのくらい縦か」をチェックしてもいい
    # if best_score is not None and best_score < vertical_ratio:
    #     return best_edge  # 縦っぽい
    # else:
    #     return None       # どれも縦っぽくない
    return best_edge



if __name__ == "__main__":
    main()