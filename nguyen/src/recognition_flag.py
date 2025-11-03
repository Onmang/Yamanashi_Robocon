# -*- coding: utf-8 -*-
# flag 認識テスト


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
    PARAM_PATH_DIS,
    PARAM_PATH_HSV,
    PARAM_HOUGH,
    PARAM_FILTER,
    PARAM_PATH_DIS_GREEN
)

DEBUG = False  # True: デバッグモードON, False: デバッグモードOFF
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MAX = 10000  # 大きすぎる塊を除外（必要に応じ調整）
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
            
            
            # 輪郭抽出は mask_morph を使うのが確実（vis を2値化しても可）
            contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 三角形検出
            triangles = detect_triangles(contours, epsilon_ratio=0.08, min_area=500)

            # 描画＆重心表示
            for tri in triangles:
                cv2.drawContours(vis, [tri], -1, (0, 0, 255), 2)   # 赤で三角形
                M = cv2.moments(tri)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)  # 重心
                    cv2.putText(vis, "Tri", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    
            if DEBUG:
                cv2.imshow("Input", overlay)
                cv2.imshow("Result", vis)
                cv2.imshow("Mask", mask)
                cv2.imshow("Mask Morph", mask_morph)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()
                
def detect_triangles(contours, epsilon_ratio=0.1, min_area=0):
    """
    輪郭リストから三角形を検出する関数
    Args:
        contours (list): cv2.findContours()で得た輪郭リスト
        epsilon_ratio (float): 輪郭近似のしきい値（arclenに対する割合）
        min_area (float): 面積の最小値（ノイズ除去用）

    Returns:
        list: 検出された三角形の輪郭リスト
    """
    approx_contours = []
    for cnt in contours:
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * arclen, True)
        if len(approx) == 3 and cv2.contourArea(approx) >= min_area:
            approx_contours.append(approx)
    return approx_contours



if __name__ == "__main__":
    main()