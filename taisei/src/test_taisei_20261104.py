#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_20260808_move_hit.py

test for moving and hitting
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
    2026-08-05

History:
    - 2026-08-05: nguyen 
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
    encode_distance_ver2,
    find_vertical_edge,
    detect_triangle_by_color,
    expand_bbox
)

from path_config import (
    PARAM_PATH_DIS_LIMIT,
    YOLO_MODEL
)

# macro
DETECTION = "detection"
ARDUINO_MOVE = "arduino_move"
HIT = "hit"
BALL = "ball"
FLAG = "flag"
POLE = "pole"
# ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
ball_idx = 4
flag_idx = 1
pole_idx = 2
bbox_color = (0, 255, 0) # green,  バウンディングボックス描画

# しきい値関係
AREA_MIN = 100  # 小ノイズ除去
AREA_MIN_FLAG = 130  # flag用三角形最小面積

# threshold
# MODE5へ移る距離。40°の認識下限173mmに近づきすぎない安全側の初期値。
# 実測表と同じ距離定義（下記dist_mm）か確認してから調整すること。
dist_th_1 = 205  # mm (仮設定。実機で停止距離・センサ位置を要確認)

# MODE1中のカメラ角度切り替え: (現在角度, 距離の閾値mm, 次の角度)
# 現在角度に対応する行だけを見るため、接近中に逆向きの切り替えは起こらない。
CAMERA_SWITCH_TABLE = (
    (0,  700,  5),
    (5,  550, 10),
    (10, 440, 15),
    (15, 350, 20),
    (20, 290, 25),
    (25, 250, 30),
    (30, 220, 35),
    (35, 200, 40),
)
CAMERA_SWITCH_CONFIRM_FRAMES = 3
CAMERA_INITIAL_ANGLE_DEG = 0
CAMERA_MOVE_TIMEOUT_S = 15.0
# LOST後はロボットを止め、5°刻みで40°まで探索してから0°方向へ戻る。
# 各位置で10フレーム観察し、3フレーム連続で有効な距離を取得したら復帰。
CAMERA_SEARCH_STEP_DEG = 5
CAMERA_SEARCH_MAX_ANGLE_DEG = 40
CAMERA_SEARCH_DWELL_FRAMES = 10
CAMERA_SEARCH_CONFIRM_FRAMES = 3
INITIAL_HIT_DISTANCE_MM = 50
HIT_MODE_TIMEOUT_S = 20.0


def next_camera_angle(dist_mm, current_angle):
    """3フレーム連続判定の1回分。角度を下げ戻す判断はしない。"""
    for angle, threshold, next_angle in CAMERA_SWITCH_TABLE:
        if current_angle == angle and 0 < dist_mm <= threshold:
            return next_angle
    return None


def make_camera_search_angles(current_angle):
    """現在角度から下向き(40°)へ探索し、その後0°まで戻る1往復の角度列。"""
    angles = list(range(CAMERA_INITIAL_ANGLE_DEG,
                        CAMERA_SEARCH_MAX_ANGLE_DEG + 1,
                        CAMERA_SEARCH_STEP_DEG))
    if current_angle not in angles:
        raise ValueError(f'Unknown camera angle for scan: {current_angle}')
    upward = [angle for angle in angles if angle > current_angle]
    downward = list(reversed(angles[:-1]))  # 40°はすでに探索済み。35→0°
    return upward + downward


def encode_camera_angle_for_arduino(angle_deg):
    """このArduinoコードのMODE2専用符号: 0=正, 1=負, 3桁角度。"""
    angle = int(angle_deg)
    if abs(angle) > 999:
        raise ValueError('MODE2 angle is outside 3-digit command range')
    return ('1' if angle < 0 else '0') + f'{abs(angle):03d}'

angle_th_1 = 1  # deg

# 打つときの閾値
hit_angle_1 = 12  # deg
hit_dis_1 = 180  # mm

# 旗を使用するかどうか
USE_FLAG = False
# 旗なしの場合の旋回角度 [deg]
FIXED_HIT_ANGLE_DEG = 30
# 追加する処理状態
MANUAL_ROTATE = "manual_rotate"
MANUAL_HIT = "manual_hit"
CAMERA_WAIT = "camera_wait"  # MODE2送信・DONE受信待ち
CAMERA_SEARCH = "camera_search"  # 車輪停止のまま、各角度でYOLO再検出
CAMERA_STOP = "camera_stop"  # MODE1停止用のMODE0を送る1周
ERROR_STOP = "error_stop"    # タイムアウト／異常停止

# arduino シリアル通信設定
ARDUINO = True #False   True:
#シリアル通信ON, False: シリアル通信OFF
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


def read_serial_lines(ser_, rx_buf):
    """非ブロッキングで受信バイトを溜め、改行まで揃った行だけを返す。

    ser は timeout=0 で開いているので readline() は改行を待たず
    「その瞬間バッファにあるバイトだけ」を返してしまう。
    115200baud では1バイト約87us間隔で届くため、ウェイト無しのループで
    ポーリングすると "DO" と "NE" に割れて DONE を取りこぼす。
    そこで bytearray に溜めてから改行単位で切り出す。

    Args:
        ser_ (serial.Serial): 開いているシリアルポート
        rx_buf (bytearray): 呼び出し側で保持し続ける受信バッファ

    Returns:
        list[str]: 改行まで揃った行のリスト（空行は除く）
    """
    n = ser_.in_waiting
    if n:
        rx_buf.extend(ser_.read(n))

    lines = []
    while True:
        idx = rx_buf.find(b"\n")
        if idx < 0:
            break
        raw = bytes(rx_buf[:idx])
        del rx_buf[: idx + 1]
        line = raw.decode("utf-8", errors="ignore").strip()
        if line:
            lines.append(line)
    return lines


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

        rotate_sent = False
        hit_sent = False
        camera_move_sent = False
        camera_move_sent_at = 0.0
        camera_target_angle = CAMERA_INITIAL_ANGLE_DEG
        camera_angle = CAMERA_INITIAL_ANGLE_DEG
        camera_switch_count = 0
        camera_move_purpose = "reset"  # reset / track / search
        camera_search_angles = []
        camera_search_next_index = 0
        camera_search_dwell_count = 0
        camera_search_found_count = 0
        hit_sent_at = 0.0
        stop_sent = False
        error_reason = ""

        # --------------------------------------------
        # メインループ
        # --------------------------------------------
        print("[DEBUG] Main loop is starting....")
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 10  # 短期見失いの許容量（フレーム数）
        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ
        # 実機: 起動直後にMODE3(50mm素振り)を一度だけ送り、DONE後MODE2(0°)。
        # Arduino未接続時はモータを動かせないため、DETECTIONから始まるテストモード。
        rasp_mode = MANUAL_HIT if ARDUINO else DETECTION
        target_obj = BALL
        initial_hit = ARDUINO
        if not ARDUINO:
            print('[SIM] ARDUINO=False: 初期素振り/カメラ原点復帰は実行されません')
            print('[SIM] カメラは実際に0度に合わせてから認識テストしてください')
        rx_buf = bytearray()  # シリアル受信バッファ（行が揃うまで溜める）
        last_sent_mode = None  # ログ抑制用：前回送ったmode

        # main loop
        while True:
            # ここにメインループの処理を記述
            state = "LOST"
            mode = 0  # arduino mode
            send_dist_mm = 0
            send_angle_deg = 0
            recog_res = False  # 検出結果初期化
            do_send = True  # 遷移した周・DONE待ちの周は送信しない

            # 場合分け
            # ----------
            # DETECTION :  ball_ball:0,  flag:1, pole:2, red_ball:3, yellow_ball:4,
            # ----------
            if rasp_mode in (DETECTION, CAMERA_SEARCH):  # 通常追跡／停止中の角度スキャン
    
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
                # best_pole_box = None
                max_ball_conf = 0.0
                max_flag_conf = 0.0
                # max_pole_conf = 0.0

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
                    # elif cls_id_ == pole_idx and conf_ > max_pole_conf:
                    #     max_pole_conf = conf_
                    #     best_pole_box = box
                        
                # 1 [Ball] check if None
                if target_obj == BALL and best_ball_box is not None:
                    # 座標・スコア・クラス取得
                    x1, y1, x2, y2 = map(int, best_ball_box.xyxy[0])
                    bst_conf = float(best_ball_box.conf[0])
                    bst_cls_id = int(best_ball_box.cls[0])
                    bst_cls_name = result.names[bst_cls_id]

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

                    # 距離が取れないフレームもMISS/探索判定に含める。
                    if cam3d is None or rob3d is None:
                        recog_res = False
                    else:
                        Xr, Yr, _ = rob3d  # [m]
                        # 正面方向に近いほど小さい
                        ang_ = compute_angles_from_position(Xr, Yr)
                        angle_str = f"{ang_:+6.1f}deg"

                        # バウンディングボックス描画
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bbox_color, 2)

                        # 描画
                        depth_mm = get_depth_at_bbox(depth_frame, x1, y1, x2, y2)
                        depth_str = f"{depth_mm:.0f}mm" if depth_mm > 0 else "N/A"

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

                # 2 [flag]
                elif target_obj == FLAG and best_flag_box is not None:

                    # 座標・スコア・クラス取得
                    bst_conf = float(best_flag_box.conf[0])
                    bst_cls_id = int(best_flag_box.cls[0])
                    bst_cls_name = result.names[bst_cls_id]

                    # 3角形をboxから見つける
                    triangles = detect_triangle_by_color(color_image, best_flag_box.xyxy[0], epsilon_ratio=0.03, area_min=AREA_MIN_FLAG, margin_ratio=0.06)

                    # バウンディングボックス描画
                    x1, y1, x2, y2 = expand_bbox(best_flag_box.xyxy[0], color_image.shape,margin_ratio=0.06)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bbox_color, 5)

                    if triangles is not None:
                        cv2.polylines(annotated_image, [triangles], isClosed=True, color=(0, 255, 255), thickness=2)

                        # 旗のポール座標を見つける
                        edge = find_vertical_edge(triangles)
                        if edge is not None:
                            p1, p2 = edge
                            mx = int((p1[0] + p2[0]) / 2)
                            my = int((p1[1] + p2[1]) / 2)

                            # result
                            recog_res = True
                            # --- 3D座標計算 ---
                            depth_image = np.asanyarray(depth_frame.get_data())
                            cam3d, rob3d = project_center_to_robot(
                                u=mx,
                                v=my,
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

                            # mx, myのdepth距離, 描画のため
                            depth_mm = 1000.0 * depth_frame.get_distance(mx, my)  # メートル単位なので1000倍する
                            depth_str = f"{depth_mm:.0f}mm" if depth_mm > 0 else "N/A"

                            # 正面方向に近いほど小さい
                            ang_ = compute_angles_from_position(Xr, Yr)
                            angle_str = f"{ang_:+6.1f}deg"

                            # label
                            label = f"{bst_cls_name} {bst_conf:.2f} | {depth_str:4}"
                            print(f"[Debug] Detected: {label} | {angle_str} | {fps_text}")
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            lx, ly = x1, max(y1 - 10, label_size[1])

                            # 描画
                            cv2.line(annotated_image, tuple(p1), tuple(p2), (0, 0, 255), 3)
                            cv2.drawMarker(annotated_image, position=(mx,my), color=(0,255,0),markerType=cv2.MARKER_CROSS, markerSize=100, thickness=5, line_type=cv2.LINE_4)
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

                if recog_res:
                    cv2.putText(
                        annotated_image, f"dist_mm: {dist_mm} mm", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                cv2.putText(
                    annotated_image,
                    f"CAM: {camera_angle} deg  SWITCH: {camera_switch_count}/{CAMERA_SWITCH_CONFIRM_FRAMES}",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
                if rasp_mode == CAMERA_SEARCH:
                    cv2.putText(
                        annotated_image,
                        f"SEARCH {camera_search_dwell_count}/{CAMERA_SEARCH_DWELL_FRAMES} "
                        f"FOUND {camera_search_found_count}/{CAMERA_SEARCH_CONFIRM_FRAMES}",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )
                
                # カラー画像表示
                cv2.imshow("YOLOv8 + RealSense", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 場合分け
            # ----------
            # ARDUINO_MOVE :  Arduinoがボールの距離を微調整
            # ----------    
            elif rasp_mode == ARDUINO_MOVE:
                # DONE待ちの間も画面とキー入力は回す（YOLO推論だけスキップ）
                color_frame, depth_frame = get_rgbd_frames(activate_cam)
                if color_frame:
                    annotated_image = np.asanyarray(color_frame.get_data()).copy()
                    cv2.putText(
                        annotated_image, "ARDUINO_MOVE : waiting DONE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                    )
                    cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # arduino調整終わったかどうかの確認。
                # 行が揃ったものだけ見る（timeout=0 の readline() は途中で返るため）
                if ARDUINO:
                    for line in read_serial_lines(ser, rx_buf):
                        print(f"[Arduino] {line}")
                        
                        if line == "DONE":
                            print("[STATE] Arduino alignment DONE")

                            if USE_FLAG:
                             # 従来どおり旗を認識する
                             rasp_mode = DETECTION
                             target_obj = FLAG

                             prev_angle = 0
                             prev_dist = 0
                             miss_count = MISS_LIMIT + 1
                            
                            else:
                             # 旗を使わず指定角度に旋回
                             rasp_mode = MANUAL_ROTATE
                             rotate_sent = False

                            do_send = False
                            break
                else:
                    # シリアル無しではDONEが来ないので待たずに次へ
                    print("[STATE] ARDUINO=False: skipping fine tuning (no Arduino responses)")
                    if USE_FLAG:
                        rasp_mode = DETECTION
                        target_obj = FLAG
                        prev_angle = 0
                        prev_dist = 0
                        miss_count = MISS_LIMIT + 1
                    else:
                        rasp_mode = MANUAL_ROTATE
                        rotate_sent = False
                    do_send = False

            elif rasp_mode == CAMERA_STOP:
                # 初回検出の周にMODE0を送る。送信失敗時は次の周に再送する。
                color_frame, _ = get_rgbd_frames(activate_cam)
                if color_frame:
                    annotated_image = np.asanyarray(color_frame.get_data()).copy()
                    cv2.putText(annotated_image, "STOP MODE1 -> CAMERA_MOVE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.imshow("YOLOv8 + RealSense", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elif rasp_mode == CAMERA_WAIT:
                # MODE2送信後はYOLO処理とMODE1送信を止めてDONEを待つ。
                color_frame, _ = get_rgbd_frames(activate_cam)
                if color_frame:
                    annotated_image = np.asanyarray(color_frame.get_data()).copy()
                    cv2.putText(
                        annotated_image, f"CAMERA_MOVE: {camera_target_angle} deg waiting DONE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )
                    cv2.imshow("YOLOv8 + RealSense", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if ARDUINO and camera_move_sent:
                    for line in read_serial_lines(ser, rx_buf):
                        print(f"[Arduino] {line}")
                        if line == "DONE":
                            camera_angle = camera_target_angle
                            camera_switch_count = 0
                            prev_angle = 0
                            prev_dist = 0
                            miss_count = MISS_LIMIT + 1
                            if camera_move_purpose == "search":
                                rasp_mode = CAMERA_SEARCH
                                camera_search_dwell_count = 0
                                camera_search_found_count = 0
                                print(f"[SEARCH] MODE2 DONE: camera={camera_angle} deg -> scan frames")
                            else:
                                rasp_mode = DETECTION
                                print(f"[STATE] MODE2 DONE: camera={camera_angle} deg -> MODE1 tracking")
                            target_obj = BALL
                            camera_move_sent = False
                            do_send = False
                            break
                        if line.startswith("CAMERA_ERROR:") or line == "CAMERA_TIMEOUT":
                            error_reason = line
                            rasp_mode = ERROR_STOP
                            stop_sent = False
                            do_send = False
                            print(f"[ERROR] MODE2 failed: {error_reason}")
                            break
                    if rasp_mode == CAMERA_WAIT and time.monotonic() - camera_move_sent_at > CAMERA_MOVE_TIMEOUT_S:
                        error_reason = f"MODE2 DONE timeout ({CAMERA_MOVE_TIMEOUT_S:.0f}s)"
                        rasp_mode = ERROR_STOP
                        stop_sent = False
                        do_send = False
                        print(f"[ERROR] {error_reason}")
                elif not ARDUINO:
                    # 通信なしでは実サーボは動かない。判定ロジックのみ疑似実行。
                    print(f"[SIM] Would send MODE2 angle={camera_target_angle} deg; no physical movement")
                    camera_angle = camera_target_angle
                    camera_switch_count = 0
                    prev_angle = 0
                    prev_dist = 0
                    miss_count = MISS_LIMIT + 1
                    if camera_move_purpose == "search":
                        rasp_mode = CAMERA_SEARCH
                        camera_search_dwell_count = 0
                        camera_search_found_count = 0
                    else:
                        rasp_mode = DETECTION
                    target_obj = BALL
                    do_send = False

            elif rasp_mode == ERROR_STOP:
                color_frame, _ = get_rgbd_frames(activate_cam)
                if color_frame:
                    annotated_image = np.asanyarray(color_frame.get_data()).copy()
                    cv2.putText(
                        annotated_image, "ERROR_STOP: " + error_reason[:48],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )
                    cv2.imshow("YOLOv8 + RealSense", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elif rasp_mode in (MANUAL_ROTATE, MANUAL_HIT):
                # Arduino待ち中もカメラ画面とqキーを有効にする
                color_frame, _ = get_rgbd_frames(activate_cam)
                if color_frame:
                    annotated_image = np.asanyarray(color_frame.get_data()).copy()
                    wait_text = (
                        "MANUAL_ROTATE : waiting DONE"
                        if rasp_mode == MANUAL_ROTATE
                        else "MANUAL_HIT : waiting DONE"
                    )
                    cv2.putText(
                        annotated_image, wait_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                    )
                    cv2.imshow("YOLOv8 + RealSense", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if rasp_mode == MANUAL_ROTATE:
                    # mode 4を送った後、旋回完了を待つ
                    if ARDUINO and rotate_sent:
                        for line in read_serial_lines(ser, rx_buf):
                            print(f"[Arduino] {line}")
                            if line == "DONE":
                                print("[STATE] ROTATE -> HIT")
                                rasp_mode = MANUAL_HIT
                                hit_sent = False
                                do_send = False  # この周にはmode 3を送らない
                                break

                elif rasp_mode == MANUAL_HIT:
                    # mode 3を送った後、Arduinoの打球完了通知を待つ
                    if ARDUINO and hit_sent:
                        if time.monotonic() - hit_sent_at > HIT_MODE_TIMEOUT_S:
                            error_reason = f"MODE3 DONE timeout ({HIT_MODE_TIMEOUT_S:.0f}s)"
                            rasp_mode = ERROR_STOP
                            stop_sent = False
                            do_send = False
                            print(f"[ERROR] {error_reason}")
                        if rasp_mode == MANUAL_HIT:
                            hit_lines = read_serial_lines(ser, rx_buf)
                        else:
                            hit_lines = []
                        for line in hit_lines:
                            print(f"[Arduino] {line}")
                            if line == "DONE":
                                if initial_hit:
                                    print("[STATE] Initial MODE3 DONE -> MODE2 0 deg")
                                else:
                                    print("[STATE] HIT DONE -> MODE2 0 deg for next ball")
                                initial_hit = False
                                # 次のボールの探索前にカメラ角度を必ず0°へ戻す。
                                rasp_mode = CAMERA_WAIT
                                camera_target_angle = CAMERA_INITIAL_ANGLE_DEG
                                camera_move_purpose = "reset"
                                camera_search_angles = []
                                camera_move_sent = False
                                camera_switch_count = 0
                                target_obj = BALL
                                prev_angle = 0
                                prev_dist = 0
                                miss_count = MISS_LIMIT + 1
                                rotate_sent = False
                                hit_sent = False
                                last_sent_mode = None
                                do_send = False
                                break
                            if line == "HIT_CANCELLED":
                                error_reason = "MODE3 cancelled"
                                rasp_mode = ERROR_STOP
                                stop_sent = False
                                do_send = False
                                print(f"[ERROR] {error_reason}")
                                break

            # ----------
            # mode選択と送信データ作成
            # ----------
            if rasp_mode == ARDUINO_MOVE:
                # mode 5は遷移時に1回送るだけ。DONE待ちでは再送しない
                do_send = False

            elif rasp_mode == CAMERA_SEARCH:
                # 車輪を停止したまま、各角度で一定フレーム数だけ再検出を試す。
                do_send = False
                camera_search_dwell_count += 1
                if state == "TRACK" and 0 < dist_mm:
                    camera_search_found_count += 1
                else:
                    camera_search_found_count = 0

                if camera_search_found_count >= CAMERA_SEARCH_CONFIRM_FRAMES:
                    rasp_mode = DETECTION  # 次の周の新しい画像でMODE1/MODE5を判定
                    camera_search_angles = []
                    camera_search_next_index = 0
                    camera_search_dwell_count = 0
                    camera_search_found_count = 0
                    camera_switch_count = 0
                    miss_count = 0
                    print(f"[SEARCH] BALL REACQUIRED at {camera_angle}deg, "
                          f"dist={dist_mm:.0f}mm -> DETECTION")
                elif camera_search_dwell_count >= CAMERA_SEARCH_DWELL_FRAMES:
                    if camera_search_next_index < len(camera_search_angles):
                        camera_target_angle = camera_search_angles[camera_search_next_index]
                        camera_search_next_index += 1
                        camera_move_purpose = "search"
                        camera_move_sent = False
                        rasp_mode = CAMERA_STOP
                        mode = 0
                        do_send = True
                        print(f"[SEARCH] not found, STOP -> MODE2 {camera_target_angle}deg "
                              f"({camera_search_next_index}/{len(camera_search_angles)})")
                    else:
                        error_reason = "CAMERA_SEARCH_EXHAUSTED (no ball within 0..40deg)"
                        rasp_mode = ERROR_STOP
                        stop_sent = False
                        print(f"[SEARCH] {error_reason}; remain stopped")

            elif rasp_mode == DETECTION and target_obj == BALL:
                if state == "TRACK":
                    # 最終的にはカメラ認識限界に達する前にMODE5へ移る。
                    # MODE5への遷移を角度変更より優先する。
                    if 0 < dist_mm <= dist_th_1:
                        mode = 5
                        send_dist_mm = 0
                        send_angle_deg = 0
                        camera_switch_count = 0
                        rasp_mode = ARDUINO_MOVE
                        print(f"[STATE] DETECTION/BALL -> ARDUINO_MOVE (dist={dist_mm}mm)")
                    else:
                        next_angle = next_camera_angle(dist_mm, camera_angle)
                        if next_angle is not None:
                            camera_switch_count += 1
                        else:
                            camera_switch_count = 0

                        if camera_switch_count >= CAMERA_SWITCH_CONFIRM_FRAMES:
                            # まずMODE0で車輪停止。次の周にMODE2を1回送る。
                            mode = 0
                            send_dist_mm = 0
                            send_angle_deg = 0
                            rasp_mode = CAMERA_STOP
                            camera_target_angle = next_angle
                            camera_move_purpose = "track"
                            camera_move_sent = False
                            camera_switch_count = 0
                            print(f"[STATE] MODE1 STOP -> MODE2 {camera_target_angle} deg (dist={dist_mm:.0f}mm)")
                        else:
                            mode = 1
                            send_dist_mm = dist_mm
                            send_angle_deg = angle_deg
                elif state == "HOLD":
                    # 画像が1フレームでも失われたら、前フレームのMODE1を保持して突進しない。
                    camera_switch_count = 0
                    mode = 0
                    send_dist_mm = 0
                    send_angle_deg = 0
                elif state == "LOST":
                    camera_switch_count = 0
                    # 一度のLOSTで有限長のスキャンを始める。毎周先頭に戻さない。
                    camera_search_angles = make_camera_search_angles(camera_angle)
                    camera_search_next_index = 1
                    camera_target_angle = camera_search_angles[0]
                    camera_move_purpose = "search"
                    camera_move_sent = False
                    camera_search_dwell_count = 0
                    camera_search_found_count = 0
                    rasp_mode = CAMERA_STOP
                    mode = 0
                    send_dist_mm = 0
                    send_angle_deg = 0
                    print(f"[SEARCH] LOST -> STOP, sweep from {camera_angle}deg; "
                          f"next MODE2={camera_target_angle}deg")

            elif rasp_mode == DETECTION and target_obj == FLAG:
                # 旗を認識して打つ方向を決める
                if state == "TRACK":
                    if abs(angle_deg) <= hit_angle_1:
                        mode = 3
                        send_dist_mm = hit_dis_1
                        send_angle_deg = 0
                        # 打球命令の送信後はMANUAL_HITで完了を待つ
                        rasp_mode = MANUAL_HIT
                        hit_sent = False
                        print(f"[STATE] DETECTION/FLAG -> HIT (angle={angle_deg}deg)")
                    else:
                        mode = 1
                        send_dist_mm = 0
                        send_angle_deg = angle_deg
                elif state == "HOLD":
                    mode = 1
                    send_dist_mm = 0
                    send_angle_deg = prev_angle
                elif state == "LOST":
                    mode = 0
                    send_dist_mm = 0
                    send_angle_deg = 0

            elif rasp_mode == CAMERA_STOP:
                # MODE0(停止)をこの周に送ってからMODE2へ移行する。
                mode = 0
                send_angle_deg = 0
                send_dist_mm = 0

            elif rasp_mode == CAMERA_WAIT:
                # MODE2の送信は1回のみ。DONE後にだけDETECTIONへ戻る。
                if not camera_move_sent:
                    mode = 2
                    send_angle_deg = camera_target_angle
                    send_dist_mm = 0
                else:
                    do_send = False

            elif rasp_mode == ERROR_STOP:
                # タイムアウト時は車輪を止め、ユーザーの停止・再起動を待つ。
                if not stop_sent:
                    mode = 0
                    send_dist_mm = 0
                    send_angle_deg = 0
                else:
                    do_send = False

            elif rasp_mode == MANUAL_ROTATE:
                # mode 4を1回だけ送信
                if not rotate_sent:
                    mode = 6
                    send_dist_mm = 0
                    send_angle_deg = FIXED_HIT_ANGLE_DEG
                else:
                    do_send = False

            elif rasp_mode == MANUAL_HIT:
                # mode 3を1回だけ送信
                if not hit_sent:
                    mode = 3
                    if initial_hit:
                        send_dist_mm = INITIAL_HIT_DISTANCE_MM
                    else:
                        send_dist_mm = 0

                    send_angle_deg = 0
                else:
                    do_send = False

            # デバッグ時は実際のMODE0送信がないので次の周にカメラ操作を疑似実行。
            if not ARDUINO and rasp_mode == CAMERA_STOP:
                rasp_mode = CAMERA_WAIT
                do_send = False

            # シリアル送信
            if ARDUINO and do_send:
                # Arduino側の現行MODE2符号変換に合わせる(正角=0, 負角=1)。
                angle_code = (
                    encode_camera_angle_for_arduino(send_angle_deg)
                    if mode == 2 else encode_angle(send_angle_deg)
                )
                dist_code = encode_distance_ver2(send_dist_mm)
                msg = f"{mode}{angle_code}{dist_code}\n"
                if len(msg.strip()) != 10:
                    raise ValueError(f"Invalid 10-digit Arduino command: {msg.strip()!r}")
                try:
                    sent_bytes = ser.write(msg.encode("ascii"))
                    if sent_bytes != len(msg):
                        raise serial.SerialTimeoutException(
                            f"Partial serial write ({sent_bytes}/{len(msg)} bytes)"
                        )
                    if mode != last_sent_mode:
                        print(f"[STATE] {rasp_mode}/{target_obj} {state} sent: {msg.strip()}")
                        last_sent_mode = mode

                    if rasp_mode == MANUAL_ROTATE and mode == 6:
                        rotate_sent = True
                    elif rasp_mode == MANUAL_HIT and mode == 3:
                        hit_sent = True
                        hit_sent_at = time.monotonic()
                    elif rasp_mode == CAMERA_STOP and mode == 0:
                        rasp_mode = CAMERA_WAIT
                        camera_move_sent = False
                    elif rasp_mode == CAMERA_WAIT and mode == 2:
                        camera_move_sent = True
                        camera_move_sent_at = time.monotonic()
                    elif rasp_mode == ERROR_STOP and mode == 0:
                        stop_sent = True
                except Exception as e:
                    print("Failed to write to serial:", e)
                    # 送れなかった場合はフラグを立てず、次の周で送信を試す

    except KeyboardInterrupt:
            print("\n===== Keyboard Interrupt =====")
            print("[FINISHED].")
            print("==============================\n")
    finally:
        # q/例外による終了時もMODE0(停止)を送ってから通信を閉じる。
        if ARDUINO and ser is not None:
            try:
                ser.write(b"0000010000\n")  # mode=0, angle=0, dist=+0
                ser.flush()
            except Exception as e:
                print(f"[WARN] Failed to send MODE0 at shutdown: {e}")
            finally:
                ser.close()
        cv2.destroyAllWindows()
        if activate_cam is not None:
            activate_cam.pipeline.stop()


if __name__ == "__main__":
    main()
