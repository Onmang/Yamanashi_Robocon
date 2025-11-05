# -*- coding: utf-8 -*-
# 実装 L字コース

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
    PARAM_HOUGH_D435I,
    PARAM_FILTER
)

# debug
DEBUG = True  # True: デバッグモードON, False: デバッグモードOFF
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
CHANGE_CAMERA_THRE_D435I = 350 # mm
CHANGE_CAMERA_THRE_D405 = 550 # mm
STOP_DIS_D405 = 100 # mm

# arduino シリアル通信設定
ARDUINO = False 
if ARDUINO:
    global ser

    serial_port = "/dev/ttyACM0"  # arduino UNO
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
    try:
        # --------------------------------------------
        # パラメータ読み込み
        # --------------------------------------------
        # hsvパラメータ読み込み
        flag_lo, flag_hi = load_hsv_from_json(PARAM_PATH_HSV[3])
        white_lo, white_hi = load_hsv_from_json(PARAM_PATH_HSV[8])
        
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
            "rx_deg": -90, 
            "ry_deg": 0,
            "rz_deg": 0,
        },
        dis_param_path=PARAM_PATH_DIS_D435I,
        hough_param_path=PARAM_HOUGH_D435I,
        ball_hsv_param_path=PARAM_PATH_HSV[2],
    )

    # RealSense D405 カメラ初期化
    # d405はcam3d
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
            dis_param_path=PARAM_PATH_DIS_D405,
            hough_param_path=PARAM_HOUGH_D405,
            ball_hsv_param_path=PARAM_PATH_HSV[9],
    )
    
    # init activate cam
    activate_cam = cam_d435i
    
    # --------------------------------------------
    # windown関係
    # --------------------------------------------
    if DEBUG:
        # 作成
        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Mask Morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        # サイズ変更
        w_re = 300
        h_re = 250
        cv2.resizeWindow("Input", w_re, h_re)
        cv2.resizeWindow("HSV Mask Morph", w_re, h_re)
        cv2.resizeWindow("Result", w_re, h_re)

    # --------------------------------------------
    # メインループ
    # --------------------------------------------
    print("[DEBUG] main loop....")
    mode = 0    # 送信フラグ
    EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
    MISS_LIMIT = 5  # 短期見失いの許容量（フレーム数）
    prev_angle = 0  # 直近の平滑化角度[deg]
    prev_dist = 0  # 直近の平滑化距離[mm]
    miss_count = 0  # 見失いカウンタ
    rasp_mode = 100  
    hitted_flag = False
    hit_ready = False
    Hit_n = 1
        
        
    except KeyboardInterrupt:
        print("[FINISHED].")
    finally:
        if ARDUINO and ser is not None:
            ser.close()
        cam_d435i.pipeline.stop()
        cam_d405.pipeline.stop()
        if DEBUG:
            cv2.destroyAllWindows()


    
if __name__ == "__main__":
    main()    # arduino シリアル通信設定