#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_taisei_20261106_ver2.py

test for moving and hitting
trace ball
"recognition_ver3.py" base

Example:
    $ ./<filename>.py  # after run: chmod +x <filename>.py

Author:
    nguyen

Date:
    2026-08-05

History:
    - 2026-08-05: nguyen 
    - 2026-11-05: taisei 
"""
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import serial
import torch
import cv2
import Jetson.GPIO as GPIO
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from rgbd_utils import (
    init_realsense_camera,
    get_rgbd_frames,
    smooth_angle_distance,
    get_depth_at_bbox,
    project_center_to_robot,
    compute_angles_from_position,
    encode_angle,
    encode_distance_ver2,
    detect_triangle_by_color,
    find_vertical_edge,
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

#非常停止ボタン
STOP_PIN = 29 #非常停止ボタンのGPIOピン

# GPIOの初期設定
GPIO.setmode(GPIO.BOARD)
GPIO.setup(STOP_PIN, GPIO.IN)

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
# ArduinoのMODE5がカメラを戻した後の角度。実機ログのangle=-10に合わせる。
# この値は状態表示用で、MODE5完了後に追加のMODE2は送らない。
CAMERA_FLAG_ANGLE_DEG = -10
# LOST後はロボットを止め、現在角度から5°刻みで40°まで探索して0°へ戻る。
# 各位置で10フレーム観察し、3フレーム連続で有効な距離を取得したら復帰。
CAMERA_SEARCH_STEP_DEG = 5
CAMERA_SEARCH_MAX_ANGLE_DEG = 40
CAMERA_SEARCH_DWELL_FRAMES = 10
CAMERA_SEARCH_CONFIRM_FRAMES = 3
SEARCH_BACKOFF_DISTANCE_MM = 100  # 最初の探索が一巡しても見つからない場合の後退量

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

# angle_th_1 = 1  # deg

# 2打目は旗を正面に合わせた姿勢を基準に、MODE7で指定角度へ旋回する。
# キーは打数管理にも使用する。1打目はMODE7を省略するため角度は使用しない。
# 正はArduinoの正旋回方向、負はその逆。実機で左右を確認すること。
SHOT_ROTATE_ANGLES_DEG = {
    1: 0,  # 打数管理用。MODE5完了後、旗合わせ・MODE7を省略してMODE3へ進む。
    2: 0,  # 旗の方向に合わせてからMODE7(0度)を送り、その向きで打つ。
    3: 0,
}

FLAG_ROTATE_MODE = 6  # 旗の探索・旗方向への旋回
SHOT_ROTATE_MODE = 7  # 打数ごとの指定角度へ旋回
# 旗がないときMODE6へ継続送信する探索角度。
FLAG_SEARCH_ANGLE_DEG_BY_SHOT = {
    1: -30,
    2: -30,
    3: -30
}
# 旗を正面と判定する角度の許容範囲 [deg]。打数ごとに設定する。
FLAG_ANGLE_TOL_DEG_BY_SHOT = {
    1: 1,
    2: 1,
    3: 1,
}
FLAG_CONFIRM_FRAMES = 3
FLAG_AREA_MIN = 130

# 打数ごとにMODE3へ送る距離 [mm]（キーは上の角度設定とそろえる）
# Arduino側の現行実装では0～100を打撃サーボ角へ変換する。
SHOT_HIT_DISTANCES_MM = {
    1: 2000,  # 一打目
    2: 2000,  # 二打目
    3: 2000,  # 三打目
}

# 追加する処理状態
MANUAL_ROTATE = "manual_rotate"
MANUAL_HIT = "manual_hit"
CAMERA_WAIT = "camera_wait"  # MODE2送信・z受信待ち
CAMERA_SEARCH = "camera_search"  # 車輪停止のまま、各角度でYOLO再検出
CAMERA_STOP = "camera_stop"  # MODE1停止用のMODE0を送る1周
SEARCH_BACKOFF = "search_backoff"  # MODE4で100mm後退し、完了後に一度だけ再探索
ERROR_STOP = "error_stop"    # タイムアウト／異常停止
ARDUINO_COMPLETE_TOKEN = "z"  # Arduinoは小文字zを改行付きで送信する

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


def read_serial_lines(ser_, rx_buf):
    """非ブロッキングで受信バイトを溜め、改行まで揃った行だけを返す。

    ser は timeout=0 で開いているので readline() は改行を待たず
    「その瞬間バッファにあるバイトだけ」を返してしまう。
    完了通知のzと改行も別々に届く場合があるため、bytearrayに溜めてから
    改行単位で切り出す。ログ中の文字を完了通知として扱わない。

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


def arduino_error(lines):
    """ERRORの後のzを成功扱いしないよう、受信した全行を先に確認する。"""
    return next((line for line in lines
                 if line.startswith(("ERROR:", "CAMERA_ERROR:"))
                 or line in ("CAMERA_TIMEOUT", "HIT_CANCELLED",
                             "ORBIT_COMMAND_TIMEOUT_STOP",
                             "Unknown numeric mode. STOP.")), None)


def debug_log(message):
    """ライブラリの出力抑止中も、制御に影響させずデバッグを表示する。"""
    try:
        print(message, file=sys.__stdout__, flush=True)
    except (OSError, ValueError):
        # ターミナルの出力失敗によって制御・送信済み判定を妨げない。
        pass


def display_mode(mode, previous_mode, context=""):
    """正常送信したMODEが変わったとき、状態と送信内容を表示する。"""
    if mode != previous_mode:
        debug_log(f"[STATE] MODE{mode}" + (f" | {context}" if context else ""))
    return mode


def quit_requested(rasp_mode, count):
    """qキーでループを終了する場合、その理由を残す。"""
    if cv2.waitKey(1) & 0xFF == ord("q"):
        debug_log(f"[FINISHED] q key received: state={rasp_mode} shot={count}")
        return True
    return False


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
    debug_log(f"[INFO] Starting to load YOLO model: {YOLO_MODEL}")
    load_started_at = time.monotonic()
    model = YOLO(YOLO_MODEL, task='detect')
    debug_log(f"[Debug] Run time for loading YOLO model: "
              f"{time.monotonic() - load_started_at:.3f} sec.")
    if ARDUINO:
        debug_log(f"[Debug] Serial Port was opened: {serial_port} baud={baud_rate}")
    else:
        debug_log("[SIM] ARDUINO=False: serial communication disabled")

    activate_cam = None 
    last_sent_mode = None
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
        debug_log("[DEBUG] Camera initialized!")

        count = 1  # 次に打つ回数。打球のzを受信したときだけ増やす。
        rotate_sent = False
        alignment_sent = False
        hit_sent = False
        camera_move_sent = False
        camera_target_angle = CAMERA_INITIAL_ANGLE_DEG
        camera_angle = CAMERA_INITIAL_ANGLE_DEG
        camera_switch_count = 0
        camera_move_purpose = "reset"  # reset / track / search
        camera_search_angles = []
        camera_search_next_index = 0
        camera_search_dwell_count = 0
        camera_search_found_count = 0
        search_backoff_used = False
        search_backoff_sent = False
        stop_sent = False
        error_reason = ""
        emergency_stop_sent = False
        flag_phase = "observe"  # observe: MODE6連続送信 / verify: 停止送信後の正面確認
        flag_fresh_frame = False
        flag_centered_count = 0

        # --------------------------------------------
        # メインループ
        # --------------------------------------------
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 10  # 短期見失いの許容量（フレーム数）
        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ
        # 起動時は直接ボール認識を始める。カメラは事前に0°へ合わせておく。
        rasp_mode = DETECTION
        target_obj = BALL
        rx_buf = bytearray()  # シリアル受信バッファ（行が揃うまで溜める）
        debug_log("[DEBUG] Main loop is starting....")
        debug_log("[STATE] Start -> DETECTION/BALL (no startup MODE2)")

        # main loop
        while True:
            if count not in SHOT_ROTATE_ANGLES_DEG:
                debug_log("[FINISHED] All configured shots completed")
                break
            if SHOT_ROTATE_ANGLES_DEG[count] is None:
                debug_log(f"[STOP] Shot {count}: rotation angle is not configured")
                break
            if SHOT_HIT_DISTANCES_MM.get(count) is None:
                debug_log(f"[STOP] Shot {count}: MODE3 distance is not configured")
                break

            # HIGH / LOW の判定
            if GPIO.input(STOP_PIN) == GPIO.LOW:

                if emergency_stop_sent:
                    # 非常停止前のzを新しい命令の完了として使わない。
                    if ARDUINO:
                        ser.reset_input_buffer()
                    rx_buf.clear()
                    emergency_stop_sent = False
                    debug_log(f"[STATE] Emergency STOP released -> "
                              f"camera={camera_angle}->{camera_target_angle}deg "
                              f"purpose={camera_move_purpose}")

                # ここにメインループの処理を記述
                state = "LOST"
                mode = 0  # arduino mode
                send_dist_mm = 0
                send_angle_deg = 0
                recog_res = False  # 検出結果初期化
                do_send = True  # 遷移した周・z待ちの周は送信しない
                flag_action = None
                flag_write_succeeded = False
                flag_frame_processed = False
                flag_angle = None

                if rasp_mode == DETECTION and target_obj == FLAG:
                    flag_lines = read_serial_lines(ser, rx_buf) if ARDUINO else []
                    for line in flag_lines:
                        debug_log(f"[Arduino] {line}")
                    failure = arduino_error(flag_lines)
                    # MODE6のzやMODE0の停止返信は待たず、異常通知だけを確認する。
                    if failure is not None:
                        error_reason = failure
                        rasp_mode = ERROR_STOP
                        stop_sent = False
                        flag_centered_count = 0
                        debug_log(f"[ERROR] {error_reason}")

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
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # FLAG・BALLともに深度マスクをかけず、元のカラー画像で推論する。
                    # ball_ball:0, flag:1, pole:2, red_ball:3, yellow_ball:4
                    results = model.predict(
                        source=color_image,
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
                        elif cls_id_ == flag_idx and conf_ > max_flag_conf:
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
                            debug_log(f"[Debug] Detected: {label} | {angle_str} | {fps_text}"
                                      f" | shot={count} cam={camera_angle}deg")
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

                    if target_obj == FLAG:
                        flag_frame_processed = True
                        if best_flag_box is not None:
                            triangle = detect_triangle_by_color(
                                color_image, best_flag_box.xyxy[0],
                                epsilon_ratio=0.03, area_min=FLAG_AREA_MIN,
                                margin_ratio=0.06,
                            )
                            if triangle is not None:
                                edge = find_vertical_edge(triangle)
                                if edge is not None:
                                    p1, p2 = edge
                                    mx = float(p1[0] + p2[0]) / 2.0
                                    # 旗の支柱位置から方向を求める。
                                    flag_angle = float(np.degrees(np.arctan2(
                                        mx - activate_cam.intr.ppx, activate_cam.intr.fx,
                                    )))
                                    recog_res = True
                                    debug_log(f"[Debug] Detected: flag {max_flag_conf:.2f}"
                                              f" | {flag_angle:+.1f}deg"
                                              f" | tol={FLAG_ANGLE_TOL_DEG_BY_SHOT[count]}deg"
                                              f" | {fps_text} | shot={count} cam={camera_angle}deg")
                                    cv2.polylines(annotated_image, [triangle], True,
                                                  (0, 255, 255), 2)
                                    cv2.line(annotated_image, tuple(p1), tuple(p2),
                                             (0, 0, 255), 3)
                                    cv2.putText(
                                        annotated_image, f"FLAG: {flag_angle:+.1f} deg",
                                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 255, 0), 2,
                                    )

                    # ボールの距離・角度と旗の方向を混ぜて平滑化しない。
                    if target_obj == FLAG and recog_res:
                        angle_deg = flag_angle
                        miss_count = 0
                        state = "TRACK"
                    elif recog_res:
                        # 初回・リセット後は測定値をそのまま採用する
                        smoothing_alpha = 1.0 if prev_dist <= 0 else EMA_ALPHA
                        angle_deg, dist_mm = smooth_angle_distance(
                            rob3d, smoothing_alpha, prev_angle, prev_dist
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
                        else:
                            state = "LOST"
                            prev_angle = 0
                            prev_dist = 0

                    if not recog_res:
                        if target_obj == BALL:
                            miss_reason = "no_yolo_box" if best_ball_box is None else "depth_missing"
                        else:
                            miss_reason = ("no_yolo_box" if best_flag_box is None
                                           else "triangle_or_edge_missing")
                        debug_log(f"[Debug] State: {state} | target={target_obj}"
                                  f" reason={miss_reason} miss={miss_count}/{MISS_LIMIT}"
                                  f" | shot={count} cam={camera_angle}deg")

                    # put text state
                    cv2.putText(
                        annotated_image, state, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                    )

                    if recog_res and target_obj == BALL:
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
                    if quit_requested(rasp_mode, count):
                        break

                # 場合分け
                # ----------
                # ARDUINO_MOVE :  Arduinoがボールの距離を微調整
                # ----------    
                elif rasp_mode == ARDUINO_MOVE:
                    alignment_done = False
                    # MODE5のzを時間制限なしで待つ。画面とキー入力は回す。
                    color_frame, depth_frame = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        cv2.putText(
                            annotated_image, "ARDUINO_MOVE : waiting z", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                        )
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                        if quit_requested(rasp_mode, count):
                            break

                    # arduino調整終わったかどうかの確認。
                    # 行が揃ったものだけ見る（timeout=0 の readline() は途中で返るため）
                    if ARDUINO and alignment_sent:
                        alignment_lines = read_serial_lines(ser, rx_buf)
                        for line in alignment_lines:
                            debug_log(f"[Arduino] {line}")
                        failure = arduino_error(alignment_lines)
                        if failure is not None:
                            error_reason = failure
                            rasp_mode = ERROR_STOP
                            stop_sent = False
                            do_send = False
                            debug_log(f"[ERROR] MODE5 failed: {error_reason}")
                        elif ARDUINO_COMPLETE_TOKEN in alignment_lines:
                            # MODE5のzは、距離調整とArduino側のカメラ復帰の完了。
                            alignment_done = True
                    elif not ARDUINO:
                        alignment_done = True

                    if alignment_done:
                        # 1打目は旗合わせとMODE7を省き、MODE5完了後に直接打つ。
                        if count == 1:
                            rasp_mode = MANUAL_HIT
                            target_obj = BALL
                            rotate_sent = False
                            hit_sent = False
                        else:
                            rasp_mode = DETECTION
                            target_obj = FLAG
                        # カメラはArduino側で復帰済み。追加のMODE2は送らない。
                        camera_angle = CAMERA_FLAG_ANGLE_DEG
                        camera_target_angle = CAMERA_FLAG_ANGLE_DEG
                        camera_move_sent = False
                        camera_switch_count = 0
                        alignment_sent = False
                        prev_angle = 0
                        prev_dist = 0
                        miss_count = MISS_LIMIT + 1
                        flag_phase = "observe"
                        flag_fresh_frame = False
                        flag_centered_count = 0
                        do_send = False
                        debug_log(f"[STATE] MODE5 z -> {rasp_mode}/{target_obj} "
                                  f"camera={camera_angle}deg shot={count}"
                                  + (" (skip FLAG and MODE7 on first shot)" if count == 1 else ""))

                elif rasp_mode == CAMERA_STOP:
                    # 初回検出の周にMODE0を送る。送信失敗時は次の周に再送する。
                    color_frame, _ = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        cv2.putText(annotated_image, "STOP MODE1 -> CAMERA_MOVE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if quit_requested(rasp_mode, count):
                        break

                elif rasp_mode == CAMERA_WAIT:
                    # MODE2送信後はYOLO処理とMODE1送信を止め、時間制限なしでzを待つ。
                    color_frame, _ = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        cv2.putText(
                            annotated_image, f"CAMERA_MOVE: {camera_target_angle} deg waiting z",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                        )
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if quit_requested(rasp_mode, count):
                        break

                    if ARDUINO and camera_move_sent:
                        camera_lines = read_serial_lines(ser, rx_buf)
                        for line in camera_lines:
                            debug_log(f"[Arduino] {line}")
                        failure = arduino_error(camera_lines)
                        if failure is not None:
                            error_reason = failure
                            rasp_mode = ERROR_STOP
                            stop_sent = False
                            do_send = False
                            debug_log(f"[ERROR] MODE2 failed: {error_reason}")
                        for line in camera_lines:
                            if failure is not None:
                                continue
                            if line == ARDUINO_COMPLETE_TOKEN:
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
                                camera_move_sent = False
                                do_send = False
                                debug_log(f"[STATE] MODE2 z: camera={camera_angle}deg"
                                          f" purpose={camera_move_purpose} -> {rasp_mode}")
                                debug_log(f"[Debug] After MODE2: miss={miss_count}/{MISS_LIMIT}"
                                          " (one invalid frame can trigger LOST)")
                                break
                    elif not ARDUINO:
                        # 通信なしでは実サーボは動かない。判定ロジックのみ疑似実行。
                        debug_log(f"[SIM] Would send MODE2 angle={camera_target_angle}deg"
                                  f" purpose={camera_move_purpose}; no physical movement")
                        camera_angle = camera_target_angle
                        camera_switch_count = 0
                        prev_angle = 0
                        prev_dist = 0
                        miss_count = MISS_LIMIT + 1
                        if camera_move_purpose == "search":
                            rasp_mode = CAMERA_SEARCH
                            camera_search_dwell_count = 0
                            camera_search_found_count = 0
                            target_obj = BALL
                        else:
                            rasp_mode = DETECTION
                            target_obj = BALL
                        do_send = False

                elif rasp_mode == SEARCH_BACKOFF:
                    # 後退中はYOLO・MODE1・MODE2を送らず、時間制限なしでzを待つ。
                    color_frame, _ = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        cv2.putText(
                            annotated_image,
                            f"MODE4 BACKOFF: -{SEARCH_BACKOFF_DISTANCE_MM} mm waiting z",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                        )
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if quit_requested(rasp_mode, count):
                        break

                    backoff_lines = (read_serial_lines(ser, rx_buf)
                                     if ARDUINO and search_backoff_sent else [])
                    for line in backoff_lines:
                        debug_log(f"[Arduino] {line}")
                    failure = arduino_error(backoff_lines)
                    if failure is not None:
                        error_reason = failure
                        rasp_mode = ERROR_STOP
                        stop_sent = False
                        debug_log(f"[ERROR] MODE4 failed: {error_reason}")
                    elif not ARDUINO or (search_backoff_sent
                                         and ARDUINO_COMPLETE_TOKEN in backoff_lines):
                        # 一巡後の現在角度(0°)から同じ探索を行う。後退済みフラグは保持する。
                        camera_search_angles = make_camera_search_angles(camera_angle)
                        camera_search_next_index = 1
                        camera_target_angle = camera_search_angles[0]
                        camera_move_purpose = "search"
                        camera_move_sent = False
                        camera_search_dwell_count = 0
                        camera_search_found_count = 0
                        camera_switch_count = 0
                        prev_angle = 0
                        prev_dist = 0
                        miss_count = MISS_LIMIT + 1
                        target_obj = BALL
                        search_backoff_sent = False
                        rasp_mode = CAMERA_STOP
                        do_send = False
                        debug_log(f"[SEARCH] MODE4 completed -> retry from {camera_angle}deg "
                                  f"toward {CAMERA_SEARCH_MAX_ANGLE_DEG}deg, then 0deg")

                elif rasp_mode == ERROR_STOP:
                    color_frame, _ = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        cv2.putText(
                            annotated_image, "ERROR_STOP: " + error_reason[:48],
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if quit_requested(rasp_mode, count):
                        break

                elif rasp_mode in (MANUAL_ROTATE, MANUAL_HIT):
                    # Arduino待ち中もカメラ画面とqキーを有効にする
                    color_frame, _ = get_rgbd_frames(activate_cam)
                    if color_frame:
                        annotated_image = np.asanyarray(color_frame.get_data()).copy()
                        wait_text = (
                            f"MODE{SHOT_ROTATE_MODE} OFFSET : waiting z"
                            if rasp_mode == MANUAL_ROTATE
                            else "MANUAL_HIT : waiting z"
                        )
                        cv2.putText(
                            annotated_image, wait_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                        )
                        cv2.imshow("YOLOv8 + RealSense", annotated_image)
                    if quit_requested(rasp_mode, count):
                        break

                    if rasp_mode == MANUAL_ROTATE:
                        # MODE7のzを時間制限なしで待ち、受信後に打つ。
                        if ARDUINO and rotate_sent:
                            rotate_lines = read_serial_lines(ser, rx_buf)
                            for line in rotate_lines:
                                debug_log(f"[Arduino] {line}")
                            failure = arduino_error(rotate_lines)
                            if failure is not None:
                                error_reason = failure
                                rasp_mode = ERROR_STOP
                                stop_sent = False
                                do_send = False
                                debug_log(f"[ERROR] MODE{SHOT_ROTATE_MODE} failed: {error_reason}")
                            elif ARDUINO_COMPLETE_TOKEN in rotate_lines:
                                rasp_mode = MANUAL_HIT
                                hit_sent = False
                                do_send = False
                                debug_log(f"[STATE] MODE{SHOT_ROTATE_MODE} z -> HIT shot={count}")
                        elif not ARDUINO:
                            debug_log(f"[SIM] SHOT OFFSET {SHOT_ROTATE_ANGLES_DEG[count]}deg -> HIT")
                            rasp_mode = MANUAL_HIT
                            hit_sent = False
                            do_send = False

                    elif rasp_mode == MANUAL_HIT:
                        # MODE3を送った後、打球完了のzを時間制限なしで待つ。
                        if ARDUINO and hit_sent:
                            hit_lines = read_serial_lines(ser, rx_buf)
                            for line in hit_lines:
                                debug_log(f"[Arduino] {line}")
                                if line == ARDUINO_COMPLETE_TOKEN:
                                    debug_log(f"[HIT] Shot {count} completed -> camera reset to 0deg")
                                    count += 1
                                    # 次のボールの探索前にカメラ角度を必ず0°へ戻す。
                                    rasp_mode = CAMERA_WAIT
                                    camera_target_angle = CAMERA_INITIAL_ANGLE_DEG
                                    camera_move_purpose = "reset"
                                    camera_search_angles = []
                                    search_backoff_used = False
                                    search_backoff_sent = False
                                    camera_move_sent = False
                                    camera_switch_count = 0
                                    target_obj = BALL
                                    prev_angle = 0
                                    prev_dist = 0
                                    miss_count = MISS_LIMIT + 1
                                    rotate_sent = False
                                    hit_sent = False
                                    do_send = False
                                    break
                                if line == "HIT_CANCELLED":
                                    error_reason = "MODE3 cancelled"
                                    rasp_mode = ERROR_STOP
                                    stop_sent = False
                                    do_send = False
                                    debug_log(f"[ERROR] {error_reason}")
                                    break

                # ----------
                # mode選択と送信データ作成
                # ----------
                if rasp_mode == ARDUINO_MOVE:
                    # 送信失敗なら再試行し、成功後だけz待ちにする。
                    if not alignment_sent:
                        mode = 5
                        send_angle_deg = 0
                        send_dist_mm = 0
                    else:
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
                        search_backoff_used = False
                        search_backoff_sent = False
                        camera_switch_count = 0
                        miss_count = 0
                        debug_log(f"[SEARCH] BALL REACQUIRED at {camera_angle}deg, "
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
                            debug_log(f"[SEARCH] not found, STOP -> MODE2 {camera_target_angle}deg "
                                      f"({camera_search_next_index}/{len(camera_search_angles)})"
                                      f" pass={2 if search_backoff_used else 1}")
                        elif not search_backoff_used:
                            # 最初の探索失敗時だけ後退する。再探索中はこのフラグを戻さない。
                            search_backoff_used = True
                            search_backoff_sent = False
                            rasp_mode = SEARCH_BACKOFF
                            debug_log(f"[SEARCH] first sweep failed -> MODE4 backoff "
                                      f"{SEARCH_BACKOFF_DISTANCE_MM}mm; wait for z without timeout")
                        else:
                            error_reason = "CAMERA_SEARCH_EXHAUSTED_AFTER_BACKOFF"
                            rasp_mode = ERROR_STOP
                            stop_sent = False
                            mode = 0
                            send_angle_deg = 0
                            send_dist_mm = 0
                            do_send = True
                            debug_log(f"[SEARCH] {error_reason}; remain stopped")

                elif rasp_mode == DETECTION and target_obj == BALL:
                    if state == "TRACK":
                        # 最終的にはカメラ認識限界に達する前にMODE5へ移る。
                        # MODE5への遷移を角度変更より優先する。
                        if 0 < dist_mm <= dist_th_1:
                            mode = 5
                            send_dist_mm = 0
                            send_angle_deg = 0
                            camera_switch_count = 0
                            alignment_sent = False
                            rasp_mode = ARDUINO_MOVE
                            debug_log(f"[STATE] DETECTION/BALL -> ARDUINO_MOVE "
                                      f"dist={dist_mm:.0f}mm threshold={dist_th_1}mm shot={count}")
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
                                debug_log(f"[STATE] MODE1 STOP -> MODE2 {camera_target_angle}deg "
                                          f"dist={dist_mm:.0f}mm current_camera={camera_angle}deg"
                                          " purpose=track")
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
                        # 参考プログラムと同じく、現在角度から40°へ下げて0°まで戻る。
                        camera_search_angles = make_camera_search_angles(camera_angle)
                        search_backoff_used = False
                        search_backoff_sent = False
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
                        debug_log(f"[SEARCH] LOST -> STOP, camera={camera_angle}deg "
                                  f"-> MODE2 {camera_target_angle}deg purpose=search "
                                  f"miss={miss_count}/{MISS_LIMIT}")

                elif rasp_mode == DETECTION and target_obj == FLAG:
                    do_send = False
                    visible = flag_frame_processed and state == "TRACK" and flag_angle is not None
                    if not flag_frame_processed:
                        flag_centered_count = 0
                    elif flag_fresh_frame:
                        # 停止命令を送った直後の画像を飛ばし、新しい画像で確認する。
                        flag_fresh_frame = False
                        flag_centered_count = 0
                    elif visible:
                        if abs(flag_angle) <= FLAG_ANGLE_TOL_DEG_BY_SHOT[count]:
                            flag_centered_count += 1
                            if flag_phase == "verify":
                                # 停止後も旗が正面にあることを確かめてからMODE7へ進む。
                                if flag_centered_count >= FLAG_CONFIRM_FRAMES:
                                    rasp_mode = MANUAL_ROTATE
                                    rotate_sent = False
                                    debug_log(f"[FLAG] Reference confirmed -> "
                                              f"offset {SHOT_ROTATE_ANGLES_DEG[count]}deg shot={count}")
                            else:
                                # 確認中も最新角度をMODE6で送り、指令切れを防ぐ。
                                flag_action = ("stop" if flag_centered_count >= FLAG_CONFIRM_FRAMES
                                               else "align")
                        else:
                            flag_centered_count = 0
                            flag_action = "align"
                    else:
                        flag_centered_count = 0
                        if state == "HOLD":
                            # 見失った直後は古い旗角度で回り続けず、一度停止する。
                            flag_action = "hold"
                        elif state == "LOST":
                            # 旗を再認識するまで、打数に応じた方向へ探索を続ける。
                            flag_action = "search"
                    if flag_action is not None:
                        mode = 0 if flag_action in ("stop", "hold") else FLAG_ROTATE_MODE
                        if flag_action == "align":
                            send_angle_deg = int(round(flag_angle))
                        elif flag_action == "search":
                            # 旗を見失ったときは、打数に対応する方向へ探索する。
                            send_angle_deg = FLAG_SEARCH_ANGLE_DEG_BY_SHOT[count]
                        else:
                            send_angle_deg = 0
                        send_dist_mm = 0
                        do_send = True
                        debug_log(f"[FLAG] shot={count} {flag_action}: MODE{mode}"
                                  f" angle={send_angle_deg}deg phase={flag_phase}"
                                  f" centered={flag_centered_count}/{FLAG_CONFIRM_FRAMES}"
                                  f" tol={FLAG_ANGLE_TOL_DEG_BY_SHOT[count]}deg")

                elif rasp_mode == CAMERA_STOP:
                    # MODE0(停止)をこの周に送ってからMODE2へ移行する。
                    mode = 0
                    send_angle_deg = 0
                    send_dist_mm = 0

                elif rasp_mode == CAMERA_WAIT:
                    # MODE2の送信は1回のみ。z後にだけDETECTIONへ戻る。
                    if not camera_move_sent:
                        mode = 2
                        send_angle_deg = camera_target_angle
                        send_dist_mm = 0
                    else:
                        do_send = False

                elif rasp_mode == SEARCH_BACKOFF:
                    if not search_backoff_sent:
                        mode = 4
                        send_angle_deg = 0
                        send_dist_mm = -SEARCH_BACKOFF_DISTANCE_MM
                    else:
                        do_send = False

                elif rasp_mode == ERROR_STOP:
                    # Arduinoの異常通知・エラー時は停止し、停止操作・再起動を待つ。
                    if not stop_sent:
                        mode = 0
                        send_dist_mm = 0
                        send_angle_deg = 0
                    else:
                        do_send = False

                elif rasp_mode == MANUAL_ROTATE:
                    # 2打目は旗方向を基準にMODE7を1回送信する。
                    if not rotate_sent:
                        rotate_angle = SHOT_ROTATE_ANGLES_DEG[count]
                        # 0度でもMODE7を送り、Arduinoのzを受信してからMODE3へ進む。
                        mode = SHOT_ROTATE_MODE
                        send_dist_mm = 0
                        # encode_angle()とArduinoの符号が逆なので、ここだけ反転する。
                        send_angle_deg = -rotate_angle
                    else:
                        do_send = False

                elif rasp_mode == MANUAL_HIT:
                    # mode 3を1回だけ送信
                    if not hit_sent:
                        mode = 3
                        send_dist_mm = SHOT_HIT_DISTANCES_MM[count]
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
                        if rasp_mode == SEARCH_BACKOFF and mode == 4:
                            # 直前のMODE2の返信をMODE4の完了として使わない。
                            ser.reset_input_buffer()
                            rx_buf.clear()
                        sent_bytes = ser.write(msg.encode("ascii"))
                        if sent_bytes != len(msg):
                            raise serial.SerialTimeoutException(
                                f"Partial serial write ({sent_bytes}/{len(msg)} bytes)"
                            )
                        last_sent_mode = display_mode(
                            mode, last_sent_mode,
                            f"shot={count} {rasp_mode}/{target_obj} {state}"
                            f" angle={send_angle_deg}deg dist={send_dist_mm}mm"
                            f" camera={camera_angle}->{camera_target_angle}deg"
                            f" purpose={camera_move_purpose} sent: {msg.strip()}",
                        )

                        flag_write_succeeded = flag_action is not None
                        if rasp_mode == SEARCH_BACKOFF and mode == 4:
                            search_backoff_sent = True
                        elif rasp_mode == ARDUINO_MOVE and mode == 5:
                            alignment_sent = True
                            debug_log("[WAIT] MODE5 completion z (no timeout)")
                        elif rasp_mode == MANUAL_ROTATE and mode == SHOT_ROTATE_MODE:
                            rotate_sent = True
                            debug_log(f"[WAIT] MODE{SHOT_ROTATE_MODE} completion z "
                                      "-> MODE3 after completion (no timeout)")
                        elif rasp_mode == MANUAL_HIT and mode == 3:
                            hit_sent = True
                            debug_log("[WAIT] MODE3 completion z (no timeout)")
                        elif rasp_mode == CAMERA_STOP and mode == 0:
                            rasp_mode = CAMERA_WAIT
                            camera_move_sent = False
                        elif rasp_mode == CAMERA_WAIT and mode == 2:
                            camera_move_sent = True
                        elif rasp_mode == ERROR_STOP and mode == 0:
                            stop_sent = True
                    except Exception as e:
                        debug_log(f"[ERROR] Failed to write to serial: {e}")
                        # 送れなかった場合はフラグを立てず、次の周で送信を試す

                if flag_action is not None and do_send and (not ARDUINO or flag_write_succeeded):
                    # MODE6の送信後は待機せず、次のフレームでも最新角度を送る。
                    if flag_action == "stop":
                        # MODE0を送信できたら、返信を待たず画像で正面を再確認する。
                        flag_phase = "verify"
                        flag_centered_count = 0
                        flag_fresh_frame = True
                    else:
                        flag_phase = "observe"
                        flag_fresh_frame = False

            else:
                if not emergency_stop_sent:
                    debug_log(f"[STOP] Emergency input active; MODE0, "
                              f"camera will reset to {CAMERA_INITIAL_ANGLE_DEG}deg after release")
                    if ARDUINO:
                        try:
                            sent_bytes = ser.write(b"0000010000\n")
                            if sent_bytes == 11:
                                emergency_stop_sent = True
                                last_sent_mode = display_mode(0, last_sent_mode, "emergency STOP")
                        except Exception as e:
                            debug_log(f"[WARN] Emergency MODE0 failed: {e}")
                    else:
                        emergency_stop_sent = True
                count = 1  # 打球カウントを初期化し、次は1打目から再開
                prev_angle = 0  # 直近の平滑化角度[deg]
                prev_dist = 0  # 直近の平滑化距離[mm]
                miss_count = 0  # 見失いカウンタ
                # 起動前にSTOPされていても、0°なら解除後すぐボール認識を始める。
                # 接近中などでカメラが傾いていた場合は0°へ戻してから再開する。
                rasp_mode = (CAMERA_WAIT
                             if ARDUINO and camera_angle != CAMERA_INITIAL_ANGLE_DEG
                             else DETECTION)
                target_obj = BALL
                camera_target_angle = CAMERA_INITIAL_ANGLE_DEG
                camera_move_purpose = "reset"
                camera_move_sent = False
                camera_switch_count = 0
                camera_search_angles = []
                search_backoff_used = False
                search_backoff_sent = False
                alignment_sent = False
                rotate_sent = False
                hit_sent = False
                stop_sent = False
                flag_phase = "observe"
                flag_fresh_frame = False
                flag_centered_count = 0
                # 停止中の返信は捨て、解除後の命令に持ち越さない。
                if ARDUINO:
                    read_serial_lines(ser, rx_buf)
                rx_buf.clear()
                time.sleep(0.01)



    except KeyboardInterrupt:
        debug_log("[FINISHED] Keyboard Interrupt")
    except Exception as e:
        debug_log(f"[ERROR] Program aborted: {type(e).__name__}: {e}"
                  f" | state={locals().get('rasp_mode', 'initializing')}"
                  f" shot={locals().get('count', '?')}")
        debug_log(traceback.format_exc())
        raise
    finally:
        # q/例外による終了時もMODE0(停止)を送ってから通信を閉じる。
        debug_log(f"[FINISHED] Leaving main -> shutdown STOP"
                  f" | state={locals().get('rasp_mode', 'initializing')}"
                  f" shot={locals().get('count', '?')}")
        if ARDUINO and ser is not None:
            try:
                sent_bytes = ser.write(b"0000010000\n")  # mode=0, angle=0, dist=+0
                ser.flush()
                if sent_bytes == 11:
                    last_sent_mode = display_mode(0, last_sent_mode, "shutdown STOP")
            except Exception as e:
                debug_log(f"[WARN] Failed to send MODE0 at shutdown: {e}")
            finally:
                ser.close()
        cv2.destroyAllWindows()
        if activate_cam is not None:
            activate_cam.pipeline.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    # ライブラリの通常出力を抑え、debug_logの出力だけターミナルへ送る。
    LOGGER.disabled = True
    with open(os.devnull, "w") as console_output:
        with redirect_stdout(console_output), redirect_stderr(console_output):
            main()
