# -*- coding: utf-8 -*-
# 実装 L字コース

import time

import cv2
import numpy as np
import serial
import RPi.GPIO as GPIO  # GPIO用のモジュールをインポート

from common_function import (
    encode_angle,
    init_realsense_camera,
    load_filter_distance_from_json,
    load_filter_params_from_json,
    load_hsv_from_json,
    project_center_to_robot,
    change_camera,
    get_rgbd_images,
    preprocess_depth_and_hsv,
    circularity_and_hough,
    evaluate_circle_detection,
    smooth_angle_distance,
    excute_state_LOST_100,
    evaluate_triangles_detection,
    check_path_safety,
    detect_triangles,
    find_best_horizontal,
    check_goal_path,
    encode_distance_ver2,
    detect_goal_post,
    PARAM_PATH_DIS_D435I,
    PARAM_PATH_HSV,
    PARAM_PATH_DIS_D405,
    PARAM_PATH_DIS_GREEN,
    PARAM_HOUGH_D405,
    PARAM_HOUGH_D435I,
    PARAM_FILTER,
)

# GPIO PIN
STOP_PIN = 23  # GPIO pin for stop signal
GPIO.setmode(GPIO.BCM)  # GPIOのモードを"GPIO.BCM"に設定
# GPIO23を入力モードに設定
GPIO.setup(STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# debug
NEKO = False  # True: ネコ表示ON, False: ネコ表示OFF
if NEKO:
    # --- 上のほう（main() の前とか）に置く ---
    def init_neko_window():
        """顔表示用ウィンドウを最初に作成"""
        cv2.namedWindow("NEKO_FACE", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("NEKO_FACE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show_neko(name=None):
        """
        画像を差し替えるだけ。waitKeyは呼ばない。
        """
        global _last_neko_name

        if name is None:
            return

        if name != _last_neko_name:
            img = NEKO_IMAGES.get(name)
            if img is None:
                print(f"[NEKO] image not found: {name}")
                return
            cv2.imshow("NEKO_FACE", img)
            _last_neko_name = name

    # ネコ表情画像の読み込み（起動時に1回だけ）
    NEKO_DIR = "./neko_img"
    NEKO_IMAGES = {
        "angry":  cv2.imread(f"{NEKO_DIR}/neko_angury.png"),
        "confuse": cv2.imread(f"{NEKO_DIR}/neko_comfuse.png"),
        "happy":  cv2.imread(f"{NEKO_DIR}/neko_happy.png"),
        "smile":  cv2.imread(f"{NEKO_DIR}/neko_smile.png"),
    }
    _last_neko_name = "happy"   # 直前に表示した名前を覚える
    init_neko_window()  # 最初に1回だけ
    cv2.waitKey(30)  # 初回だけちょっと待って描画させる
    show_neko("happy")


# windname
WINDOW_INPUT = "Input"
WINDOW_MASK = "HSV Mask Morph"
WINDOW_RESULT = "Result"

# しきい値関係
CIRC_MIN = 0.80
AREA_MIN = 100  # 小ノイズ除去
AREA_MIN_FLAG = 120  # flag用三角形最小面積
AREA_MIN_POST = 150  # post用三角形最小面積
ALPHA_VAL = 0.8  # 安全pathマージン
CHANGE_CAMERA_THRE_D435I = 600  # mm
CHANGE_CAMERA_THRE_D405 = CHANGE_CAMERA_THRE_D435I + 150  # mm
dist_th_1 = 70  #70  # mm, plan B : 200
dist_move = 70  # mm ボール打つ準備時の移動距離
angle_th_1 = 1  # deg

# plan B: d405で認識できない場合に使うモード
PLAN_B = False
if PLAN_B:
    dist_th_1 = 200  # mm
    dist_plan_b = 130  # mm
    
# ボール色指定
if False: # 青いボール
    ball_d435i = 2
    ball_d405  = 9
else: # 黄色ボール
    ball_d435i = 1
    ball_d405  = 1
    

# ゴール認識閾値 、2打目以上、通常
dist_min_cm_goal = 30  # cm

dist_max_cm_goal = 150  # cm
# ゴール認識閾値 、ゴールが近すぎる場合
dist_min_cm_goal_2 = 10  # cm
dist_max_cm_goal_2 = 80  # cm

# わからないときの回転角度
angle_if_lost = 10  # deg

# わからないときの移動距離
dist_if_lost = 100

# 打つときの閾値
hit_angle_1 = 8  # deg
hit_dis_1 = 1500  # mm
hit_delay_time = 10.0  # sec 打つ動作の待機時間

# arduino シリアル通信設定
ARDUINO = True  # True: シリアル通信ON, False: シリアル通信OFF
if ARDUINO:
    global ser
    serial_port = "/dev/ttyACM0"  # arduino UNO
    # serial_port = "/dev/ttyUSB0"  # arduino UNO
    
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
    cam_d435i = None
    cam_d405 = None
    # --------------------------------------------
    # windown関係
    # --------------------------------------------
    # if DEBUG:
    #     # 作成
    #     cv2.namedWindow(WINDOW_INPUT, cv2.WINDOW_NORMAL)
    #     cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    #     cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)

    #     # サイズ変更
    #     w_re = 300
    #     h_re = 250
    #     cv2.resizeWindow(WINDOW_INPUT, w_re, h_re)
    #     cv2.resizeWindow(WINDOW_MASK, w_re, h_re)
    #     cv2.resizeWindow(WINDOW_RESULT, w_re, h_re)

    try:
        # --------------------------------------------
        # パラメータ読み込み
        # --------------------------------------------
        # hsvパラメータ読み込み
        flag_lo, flag_hi = load_hsv_from_json(PARAM_PATH_HSV[3])
        white_lo, white_hi = load_hsv_from_json(PARAM_PATH_HSV[8])
        post_lo, post_hi = load_hsv_from_json(PARAM_PATH_HSV[10])

        # ガウシアンフィルター
        gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)

        # 距離フィルタのデフォルト値
        green_dist_min_cm, green_dist_max_cm = load_filter_distance_from_json(
            PARAM_PATH_DIS_GREEN
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
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
            dis_param_path=PARAM_PATH_DIS_D435I,
            hough_param_path=PARAM_HOUGH_D435I,
            ball_hsv_param_path=PARAM_PATH_HSV[ball_d435i],
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
            ball_hsv_param_path=PARAM_PATH_HSV[ball_d405],
        )

        # init activate cam
        time.sleep(1.0)  # カメラ安定化待ち
        print("[DEBUG] Camera initialized.")
        activate_cam = cam_d405

        # --------------------------------------------
        # メインループ
        # --------------------------------------------
        print("[DEBUG] main loop....")
        EMA_ALPHA = 0.30  # 0.1～0.5 で調整（大きいほど追従が速い／ノイズに弱い）
        MISS_LIMIT = 10  # 短期見失いの許容量（フレーム数）
        prev_angle = 0  # 直近の平滑化角度[deg]
        prev_dist = 0  # 直近の平滑化距離[mm]
        miss_count = 0  # 見失いカウンタ
        rasp_mode = 100
        hit_ready = False
        Hit_n = 1
        delay_after_hit = False

        # main loop
        while True:
            # 緊急停止入力
            if GPIO.input(STOP_PIN) == GPIO.HIGH:  # GPIO23が"1"のとき
                # print("[Debug] Emergency Stop Activated!")
                Hit_n = 1
                rasp_mode = 100
                if activate_cam is not cam_d405:
                    activate_cam = cam_d405
                if NEKO: show_neko("confuse")
                continue

            # ここにメインループの処理を記述
            state = "LOST"
            mode = 0
            send_dist_mm = 0
            send_angle_deg = 0
            recog_res = False  # 検出結果初期化

            # 画像取得, すべてのモード共通
            color_image, depth_image = get_rgbd_images(activate_cam)

            # フレームがまだ来てなかったら次のループへ
            if color_image is None or depth_image is None:
                continue
            # 場合分け
            # ----------
            # 100: ボール探索モード
            # ----------
            match rasp_mode:
                ### ボール探索モード ###
                case 100:
                    # 前処理の2値化
                    mask_morph, vis = preprocess_depth_and_hsv(
                        color_image,
                        depth_image,
                        activate_cam,
                        dist_min_cm=None,
                        dist_max_cm=None,
                        gaus_k=gaus_k,
                        sigmaX=sigmaX,
                        hsv_lo=activate_cam.ball_lo,
                        hsv_hi=activate_cam.ball_hi,
                    )
                    # 円検出
                    circles = circularity_and_hough(
                        mask_morph,
                        vis,
                        activate_cam,
                        area_min=AREA_MIN,
                        circ_min=CIRC_MIN,
                    )

                    # 検出の評価
                    recog_res, c_x, c_y, c_r, cam3d, rob3d = evaluate_circle_detection(
                        circles, depth_image, activate_cam
                    )

                    # # 3打目以降はpath確認
                    # if recog_res and rob3d[2] < 1000 and Hit_n >= 3:
                    #     # 前処理の2値化
                    #     mask_morph_add, vis = preprocess_depth_and_hsv(
                    #         color_image,
                    #         depth_image,
                    #         activate_cam,
                    #         dist_min_cm=None,
                    #         dist_max_cm=None,
                    #         gaus_k=gaus_k,
                    #         sigmaX=sigmaX,
                    #         hsv_lo=white_lo,
                    #         hsv_hi=white_hi,
                    #     )

                    #     recog_res, cam3d, rob3d = check_path_safety(
                    #         mask_morph_add, depth_image, activate_cam, alpha=ALPHA_VAL
                    #     )

                    #     # --- ロボット位置（仮） ---
                    #     h, w = vis.shape[:2]
                    #     robot_xy = (w // 2, h - 10)

                    #     # ゴールpath確認
                    #     path_clear, dist = check_goal_path(
                    #         mask_morph_add,
                    #         robot_xy=robot_xy,
                    #         goal_xy=(c_x, c_y),
                    #         alpha=ALPHA_VAL,
                    #     )
                    #     # --- 角度で探す代替ルート ---
                    #     if not path_clear:
                    #         best_h_pt, _ = find_best_horizontal(
                    #             dist_img=dist,
                    #             origin=robot_xy,
                    #             goal_y=c_y,
                    #             x_step=5,
                    #             step_along_line=3,
                    #             min_safe_dist=10.0,
                    #         )
                    #         # --- 3D座標計算 ---
                    #         cam3d, rob3d = project_center_to_robot(
                    #             u=best_h_pt[0],
                    #             v=best_h_pt[1],
                    #             depth_image=depth_image,
                    #             depth_scale=activate_cam.depth_scale,
                    #             intr=activate_cam.intr,
                    #             T_cam2rob=activate_cam.T_cam2rob,
                    #             roi=7,
                    #         )
                ### ゴール探索モード ###
                case 200:
                    if activate_cam is not cam_d435i:
                        activate_cam = cam_d435i
                    # 前処理の2値化
                    if Hit_n == 1:
                        mask_morph, vis = preprocess_depth_and_hsv(
                            color_image,
                            depth_image,
                            activate_cam,
                            dist_min_cm=None,
                            dist_max_cm=None,
                            gaus_k=gaus_k,
                            sigmaX=sigmaX,
                            hsv_lo=flag_lo,
                            hsv_hi=flag_hi,
                        )
                    else:
                        mask_morph, vis = preprocess_depth_and_hsv(
                            color_image,
                            depth_image,
                            activate_cam,
                            dist_min_cm=dist_min_cm_goal,
                            dist_max_cm=dist_max_cm_goal,
                            gaus_k=gaus_k,
                            sigmaX=sigmaX,
                            hsv_lo=flag_lo,
                            hsv_hi=flag_hi,
                        )
                    # 三角形検出
                    triangles = detect_triangles(
                        mask_morph,
                        area_min_label=AREA_MIN,
                        area_min=AREA_MIN_FLAG,
                        epsilon_ratio=0.08,
                    )

                    # 三角形を選別
                    recog_res, rob3d, cam3d, goal_xy_pixel = (
                        evaluate_triangles_detection(
                            triangles, depth_image, activate_cam
                        )
                    )

                ## ゴールが近すぎる場合 ###
                case 210:
                    if activate_cam is not cam_d435i:
                        activate_cam = cam_d435i

                    mask_morph, vis = preprocess_depth_and_hsv(
                        color_image,
                        depth_image,
                        activate_cam,
                        dist_min_cm=dist_min_cm_goal_2,
                        dist_max_cm=dist_max_cm_goal_2,
                        gaus_k=gaus_k,
                        sigmaX=sigmaX,
                        hsv_lo=post_lo,
                        hsv_hi=post_hi,
                    )

                    recog_res, _, rob3d, _, _ = detect_goal_post(
                        mask_morph,
                        depth_image,
                        activate_cam,
                        area_min_label=AREA_MIN,
                        area_min_post=AREA_MIN_POST,
                        aspect_min=6.0,
                        approx_eps_ratio=0.03,
                    )

                ### 打つ　###
                case 300:
                    print(f"[Debug] case {rasp_mode}: Hitting the ball...")
                    pass
                ### 自由歩きモード ###
                case 400:
                    if activate_cam is not cam_d435i:
                        activate_cam = cam_d435i

                    # 前処理の2値化
                    mask_morph, vis = preprocess_depth_and_hsv(
                        color_image,
                        depth_image,
                        activate_cam,
                        dist_min_cm=green_dist_min_cm,
                        dist_max_cm=green_dist_max_cm,
                        gaus_k=gaus_k,
                        sigmaX=sigmaX,
                        hsv_lo=white_lo,
                        hsv_hi=white_hi,
                    )

                    recog_res, cam3d, rob3d = check_path_safety(
                        mask_morph, depth_image, activate_cam, alpha=ALPHA_VAL
                    )

                ### デフォルト ###
                case _:
                    print("[Debug] Undefined Mode in step 1")
                    continue

            # 検出成功
            if recog_res:
                # 距離を平均化
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
                if NEKO: show_neko("smile")
            else:
                # 見失いカウンタ増加
                miss_count += 1
                if miss_count <= MISS_LIMIT:
                    state = "HOLD"
                else:
                    state = "LOST"
                    prev_angle = 0
                    prev_dist = 0

            # 場合分け
            # ----------
            # 100: ボール探索モード
            # ----------
            match rasp_mode:
                ### ボール探索モード ###
                case 100:
                    match state:
                        case "TRACK":
                            # カメラ変更判定
                            _, _, Zc = cam3d
                            activate_cam = change_camera(
                                activate_cam,
                                cam_d435i,
                                cam_d405,
                                Zc * 1000,
                                thre_d435i=CHANGE_CAMERA_THRE_D435I,
                                thre_d405=CHANGE_CAMERA_THRE_D405,
                            )
                            if dist_mm < dist_th_1:
                                print("[Debug] dist_mm < dist_th_1, set to rasp_mode 200")
                                mode = 0
                                send_dist_mm = 0
                                send_angle_deg = 0    
                                rasp_mode = 200
                                if PLAN_B:
                                    ## plan B ##
                                    ## d405で認識できない場合の代替ルート
                                    mode = 4
                                    send_dist_mm = dist_plan_b
                                    send_angle_deg = 0
                                    time.sleep(1.0)
                            else:
                                mode = 1
                                send_dist_mm = dist_mm
                                send_angle_deg = angle_deg     
            
                        case "HOLD":
                            mode = 1
                            send_dist_mm = prev_dist
                            send_angle_deg = prev_angle
                        case "LOST":
                            print(f"[Debug] case {rasp_mode} in step 2: Lost ball...")
                            mode = 4
                            send_dist_mm = dist_if_lost
                            send_angle_deg = 0
                            if NEKO: show_neko("confuse")   # ここで表示
                ### ゴールに向く ###
                case 200:
                    match state:
                        case "TRACK":
                            if abs(angle_deg) <= angle_th_1:  # 角度閾値OK
                                ###############
                                # コース確認必要
                                ###############
                                mask_morph, vis = preprocess_depth_and_hsv(
                                    color_image,
                                    depth_image,
                                    activate_cam,
                                    dist_min_cm=None,
                                    dist_max_cm=None,
                                    gaus_k=gaus_k,
                                    sigmaX=sigmaX,
                                    hsv_lo=white_lo,
                                    hsv_hi=white_hi,
                                )

                                if Hit_n == 1:
                                    mode = 4
                                    send_dist_mm = 0
                                    send_angle_deg = hit_angle_1
                                else:
                                    mode = 0
                                    send_dist_mm = 0
                                    send_angle_deg = 0
                                rasp_mode = 300

                            else:  # まだ
                                mode = 1
                                send_dist_mm = 0
                                send_angle_deg = angle_deg
                        case "HOLD":
                            mode = 1
                            send_dist_mm = 0
                            send_angle_deg = prev_angle
                        case "LOST":
                            print(f"[Debug] case {rasp_mode} in step 2: Lost Goal...")
                            if NEKO: show_neko("angury")
                            # mode = 3  # 超音波探索
                            # send_dist_mm = 0
                            # send_angle_deg = 0
                            mode = 4  
                            send_dist_mm = 0
                            send_angle_deg = -20
                            if (
                                miss_count > FPS * 10
                            ):  # 10秒以上見失ったらボール探索へ戻る
                                rasp_mode = 210
                case 300:
                    if hit_ready:
                        print(f"[Debug] case {rasp_mode} in step 2: Ready to hit ball : order {Hit_n} !!!")
                        # 送信
                        mode = 2
                        send_dist_mm = hit_dis_1
                        send_angle_deg = 0
                        delay_after_hit = True

                        # 初期化
                        prev_angle = 0
                        prev_dist = 0
                        miss_count = 0

                        # 更新
                        Hit_n += 1
                        hit_ready = False
                        activate_cam = cam_d435i
                        rasp_mode = 100
                        if NEKO: show_neko("happy")
                        time.sleep(1.0)

                    else:
                        # 送信
                        mode = 4
                        send_dist_mm = -dist_move  # 70mm バック
                        send_angle_deg = 0
                        time.sleep(0.5)
                        
                        # 打つ準備            
                        hit_ready = True
                case 400:
                    match state:
                        case "TRACK":
                            mode = 1
                            send_dist_mm = dist_mm
                            send_angle_deg = angle_deg
                        case "HOLD":
                            mode = 1
                            send_dist_mm = prev_dist
                            send_angle_deg = prev_angle
                        case "LOST":
                            mode = 1  # もうわからない
                            send_dist_mm = 0
                            send_angle_deg = -angle_if_lost
                case 210:  # ゴールが近い過ぎるときとか
                    match state:
                        case "TRACK":
                            if abs(angle_deg) <= angle_th_1:
                                mode = 1
                                send_dist_mm = dist_mm
                                send_angle_deg = angle_deg
                                activate_cam = cam_d405
                                rasp_mode = 300
                            else:
                                mode = 1
                                send_dist_mm = 0
                                send_angle_deg = angle_deg
                        case "HOLD":
                            mode = 1
                            send_dist_mm = 0
                            send_angle_deg = prev_angle
                        case "LOST":
                            mode = 1  # もうわからない
                            send_dist_mm = 0
                            send_angle_deg = -angle_if_lost
                case _:
                    print("[Debug] Undefined Mode in step 2")
                    continue

            # シリアル送信
            if ARDUINO:
                angle_code = encode_angle(send_angle_deg)
                dist_code = encode_distance_ver2(send_dist_mm)
                msg = f"{mode}{angle_code}{dist_code}\n"
                try:
                    ser.write(msg.encode("ascii"))
                    # print(f"[Debug] Sent{(state)}: {msg.strip()}")
                    if delay_after_hit:
                        print("[Debug] Delay for hitting:", hit_delay_time)
                        time.sleep(hit_delay_time)  # 打つ時間待機
                        print("[Debug] Delay finished.")
                        delay_after_hit = False

                except Exception as e:
                    print("Failed to write to serial:", e)
                    pass
            if NEKO:
                # OpenCVのイベント処理は1か所に集約する
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[DEBUG] 'q' pressed -> exit")
                    break

                
    except KeyboardInterrupt:
        print("\n===== Keyboard Interrupt =====")
        print("[FINISHED].")
        print("==============================\n")
    finally:
        GPIO.cleanup()
        if ARDUINO and ser is not None:
            ser.close()
        if cam_d435i is not None:
            cam_d435i.pipeline.stop()
        if cam_d405 is not None:
            cam_d405.pipeline.stop()
        if NEKO:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # arduino シリアル通信設定
