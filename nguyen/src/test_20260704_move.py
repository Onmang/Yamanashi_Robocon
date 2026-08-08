#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_20260704_move.py

test for move 
trace ball
"recognition_ver3.py" base

Example:
    $ ./<filename>.py  # after run: chmod +x <filename>.py

            # time measurement
            start_time = time.time()

            # end
            end_time = time.time()
            runtime = start_time-end_time
            print(f"[time]{runtime}")



Author:
    nguyen

Date:
    2026-07-04

History:
    - 2026-07-04: nguyen coped from 2025
"""
import time
import numpy as np
import serial
import torch
import cv2
from ultralytics import YOLO

from rgbd_utils import (
    init_realsense_camera,
    get_rgbd_frames,
    smooth_angle_distance,
    get_depth_at_bbox,
    project_center_to_robot,
    compute_angles_from_position,
    encode_angle,
    encode_distance_ver2
)

from path_config import (
    PARAM_PATH_DIS_LIMIT,
    YOLO_MODEL
)

# macro
DETECTION = "detection"
HIT = "hit"
BALL = "ball"
FLAG = "flag"
POLE = "pole"
# ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
ball_idx = 4
flag_idx = 1
pole_idx = 2
bbox_color = (0, 255, 0) # green,  バウンディングボックス描画

# threshold
dist_th_1 = 100  #70  # mm
angle_th_1 = 1  # deg

# arduino シリアル通信設定
ARDUINO = True  # True: シリアル通信ON, False: シリアル通信OFF
if ARDUINO:
    global ser
    serial_port = "/dev/ttyUSB0"  # arduino UNO, ttyACM0
    
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
    print("[Debug] Serial Port was opened:", serial_port)


def main():
    """
    flowchart
    detect ball 
    move (2025 code)

    """
    # yolo paramas
    conf = 0.5 # 信頼度閾値 (0~1)
    iou = 0.45 # IoU閾値

    # モデル読み込み
    print(f"[INFO] Starting to load YOLO model: {YOLO_MODEL}")
    s_time = time.time()
    model = YOLO(YOLO_MODEL, task='detect')
    e_time = time.time()
    run_time = e_time - s_time
    print(f"[Debug] Run time for loading YOLO model: {run_time:.3f} sec.")

    activate_cam = None 
    try:

        # --------------------------------------------
        # カメラ初期化
        # --------------------------------------------
        # 解像度とFPS
        W, H, FPS = 640, 480, 30

        # RealSense カメラ初期化
        activate_cam = init_realsense_camera(
            name="d435i",
            serial="949122070535",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": -(32.5 * 0.001),  # m
                "ty": 0,  # m, -50 * 0.001
                "tz": 0,  # m, 200 * 0.001
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
            dis_param_path=PARAM_PATH_DIS_LIMIT,
        )

        # for debug
        # activate_cam = init_realsense_camera(
        #     name="d405",
        #     serial="218622274519",  # 実機のシリアル
        #     width=W,
        #     height=H,
        #     fps=FPS,
        #     extrinsic_guess={
        #         "tx": 0,  # m
        #         "ty": 0,  # m
        #         "tz": 0,  # m
        #         "rx_deg": -90,
        #         "ry_deg": 0,
        #         "rz_deg": 0,
        #     },
        #     dis_param_path=PARAM_PATH_DIS_LIMIT,
        # )

        # init activate cam
        time.sleep(1.0)  # カメラ安定化待ち
        print("[DEBUG] Camera initialized!")

        # --------------------------------------------
        # メインループ
        # --------------------------------------------
        print("[DEBUG] Main loop is starting....")
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 10  # 短期見失いの許容量（フレーム数）
        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ
        rasp_mode = DETECTION
        target_obj = BALL

        # main loop
        while True:
            # ここにメインループの処理を記述
            state = "LOST"
            mode = 0  # arduino mode
            send_dist_mm = 0
            send_angle_deg = 0
            recog_res = False  # 検出結果初期化

            # 場合分け
            # ----------
            # DETECTION :  ball_ball:0,  flag:1, pole:2, red_ball:3, yellow_ball:4,
            # ----------
            if rasp_mode == DETECTION: # rasp_mode == DETECTION
    
                # 画像取得, すべてのモード共通
                color_frame, depth_frame = get_rgbd_frames(activate_cam)

                # フレームがまだ来てなかったら次のループへ
                if not color_frame or not depth_frame:
                    continue
                
                # make image
                color_image = np.asanyarray(color_frame.get_data())

                # YOLOv8 推論
                # ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
                results = model.predict(
                    source=color_image,
                    #classes=[ball_idx, flag_idx, pole_idx],
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )
                result = results[0]  # this is List type

                # FPS 表示
                annotated_image = color_image.copy()
                fps_text = f"FPS: {1 / (results[0].speed['inference'] / 1000 + 1e-9):.1f}"
                cv2.putText(
                    annotated_image, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

                # best ball
                best_ball_box = None
                best_flag_box = None
                best_pole_box = None
                max_ball_conf = 0.0
                max_flag_conf = 0.0
                max_pole_conf = 0.0

                # check the all boxes
                for box in result.boxes:
                    cls_id_ = int(box.cls[0]) # class ID
                    conf_ = float(box.conf[0]) # conf.

                    # input the best conf.
                    if cls_id_ == ball_idx and conf_ > max_ball_conf:
                        max_ball_conf = conf_
                        best_ball_box = box
                    elif cls_id_== flag_idx and conf_ > max_flag_conf:
                        max_flag_conf = conf_
                        best_flag_box = box
                    elif cls_id_ == pole_idx and conf_ > max_pole_conf:
                        max_pole_conf = conf_
                        best_pole_box = box
                        
                # check if None
                if target_obj == BALL and best_ball_box is not None:

                    # 座標・スコア・クラス取得
                    x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0])
                    bst_conf = float(best_ball_box.conf[0])
                    bst_cls_id = int(best_ball_box.cls[0])
                    bst_cls_name = result.names[bst_cls_id]

                    # 深度取得
                    depth_mm = get_depth_at_bbox(depth_frame, x1, y1, x2, y2)
                    depth_str = f"{depth_mm:.0f}mm" if depth_mm > 0 else "N/A"

                    # result
                    recog_res = True
                    # --- 3D座標計算 ---
                    depth_image = np.asanyarray(depth_frame.get_data())
                    cam3d, rob3d = project_center_to_robot(
                        u=int((x1 + x2) / 2),
                        v=int((y1 + y2) / 2),
                        depth_image=depth_image,
                        depth_scale=activate_cam.depth_scale,
                        intr=activate_cam.intr,
                        T_cam2rob=activate_cam.T_cam2rob,
                        roi=7,
                    )

                    # angle deg
                    if cam3d is None or rob3d is None:
                        continue		
                    Xr, Yr, _ = rob3d  # [m]
                    # 正面方向に近いほど小さい
                    ang_ = compute_angles_from_position(Xr, Yr)
                    angle_str = f"{ang_:+6.1f}deg"


                    # バウンディングボックス描画
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bbox_color, 2)

                    # ラベル
                    label = f"{bst_cls_name} {bst_conf:.2f} | {depth_str:4}"
                    print(f"[Debug] Detected: {label} | {angle_str} | {fps_text}")
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    lx, ly = x1, max(y1 - 10, label_size[1])
                    cv2.rectangle(
                        annotated_image,
                        (lx, ly - label_size[1] - 4),
                        (lx + label_size[0], ly + 4),
                        bbox_color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                    )

                # 検出成功
                if recog_res:
                    # 距離を平均化, test:cam3d, truth:rob3d
                    angle_deg, dist_mm = smooth_angle_distance(
                        rob3d, EMA_ALPHA, prev_angle, prev_dist
                    )

                    # 前回値更新
                    prev_angle = angle_deg
                    prev_dist = dist_mm

                    # 見失いカウンタリセット
                    miss_count = 0

                    # 表示情報
                    state = "TRACK"
                else:
                    # 見失いカウンタ増加
                    miss_count += 1
                    if miss_count <= MISS_LIMIT:
                        state = "HOLD"
                        print(f"[Debug] State: {state}")
                    else:
                        state = "LOST"
                        print(f"[Debug] State: {state}")
                        prev_angle = 0
                        prev_dist = 0

                # put text state
                cv2.putText(
                    annotated_image, state, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                )
                
                # カラー画像表示
                cv2.imshow("YOLOv8 + RealSense", annotated_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # move
            if rasp_mode == DETECTION:
                if state == "TRACK":
                    if dist_mm < dist_th_1:
                        mode = 0
                        send_dist_mm = 0
                        send_angle_deg = 0    
                        rasp_mode = DETECTION
                    else:
                        mode = 1
                        send_dist_mm = dist_mm
                        send_angle_deg = angle_deg 
                        rasp_mode = DETECTION
                elif state == "HOLD":
                        mode = 1
                        send_dist_mm = prev_dist
                        send_angle_deg = prev_angle
                elif state == "LOST":
                        mode = 0
                        send_dist_mm = 0
                        send_angle_deg = 0
            elif rasp_mode == HIT:
                pass


            # シリアル送信
            if ARDUINO:
                angle_code = encode_angle(send_angle_deg)
                dist_code = encode_distance_ver2(send_dist_mm)
                msg = f"{mode}{angle_code}{dist_code}\n"
                try:
                    ser.write(msg.encode("ascii"))
                    #print(f"[Debug] Sent {(state)}: {msg.strip()}")
                    # if delay_after_hit:
                    #     print("[Debug] Delay for hitting:", hit_delay_time)
                    #     time.sleep(hit_delay_time)  # 打つ時間待機
                    #     print("[Debug] Delay finished.")
                    #     delay_after_hit = False

                except Exception as e:
                    print("Failed to write to serial:", e)
                    pass

    except KeyboardInterrupt:
            print("\n===== Keyboard Interrupt =====")
            print("[FINISHED].")
            print("==============================\n")
    finally:
        cv2.destroyAllWindows()
        if ARDUINO and ser is not None:
            ser.close()
        if activate_cam is not None:
            activate_cam.pipeline.stop()


if __name__ == "__main__":
    main()
