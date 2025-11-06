# -*- coding: utf-8 -*-
# recognition_ver2.py
# 会場で距離・HSV・ハフのパラメータを調整するツール
# 2カメラ(D435i / D405)を起動し、キー操作でどちらのカメラの設定をいじるか切り替える
# 2025/11/06 + white/poll 追加版

import json
import time
from pathlib import Path

import cv2
import numpy as np

from common_function import (
    # カメラまわり
    init_realsense_camera,
    get_rgbd_images,
    preprocess_depth_and_hsv,
    circularity_and_hough,
    evaluate_circle_detection,
    detect_triangles,
    evaluate_triangles_detection,
    # パス類
    PARAM_PATH_DIS_D435I,
    PARAM_PATH_DIS_D405,
    PARAM_PATH_HSV,
    PARAM_HOUGH_D435I,
    PARAM_HOUGH_D405,
    PARAM_FILTER,
    load_filter_params_from_json,
)

# windown names
WIN_INPUT = "Input"
WIN_HSV_MASK = "HSV Mask Morph"
WIN_RESULT = "Result"
WIN_DIST_CTRL = "Distance Control"
WIN_HSV_CTRL = "HSV Control"
WIN_HOUGH_CTRL = "Hough Control"

# ---------------------------------------------------------
# トラックバー用ダミー
# ---------------------------------------------------------
def _noop(x):
    pass

# ---------------------------------------------------------
# トラックバー作成系
# ---------------------------------------------------------
def create_distance_trackbars(win, max_dist_cm=400):
    cv2.createTrackbar("Dist_min [cm]", win, 7, max_dist_cm, _noop)
    cv2.createTrackbar("Dist_max [cm]", win, 50, max_dist_cm, _noop)

def get_distance_range(win):
    dmin = cv2.getTrackbarPos("Dist_min [cm]", win)
    dmax = cv2.getTrackbarPos("Dist_max [cm]", win)
    return dmin, dmax

def create_hsv_trackbars(win):
    cv2.createTrackbar("H_low", win, 0, 179, _noop)
    cv2.createTrackbar("H_high", win, 179, 179, _noop)
    cv2.createTrackbar("S_low", win, 0, 255, _noop)
    cv2.createTrackbar("S_high", win, 255, 255, _noop)
    cv2.createTrackbar("V_low", win, 0, 255, _noop)
    cv2.createTrackbar("V_high", win, 255, 255, _noop)

def get_hsv_range(win):
    hl = cv2.getTrackbarPos("H_low", win)
    hh = cv2.getTrackbarPos("H_high", win)
    sl = cv2.getTrackbarPos("S_low", win)
    sh = cv2.getTrackbarPos("S_high", win)
    vl = cv2.getTrackbarPos("V_low", win)
    vh = cv2.getTrackbarPos("V_high", win)
    return (hl, sl, vl), (hh, sh, vh)

def create_hough_trackbars(win):
    cv2.createTrackbar("minDist", win, 50, 500, _noop)
    cv2.createTrackbar("param1", win, 100, 300, _noop)
    cv2.createTrackbar("param2", win, 30, 150, _noop)
    cv2.createTrackbar("minRadius", win, 5, 100, _noop)
    cv2.createTrackbar("maxRadius", win, 80, 200, _noop)

def get_hough_params(win):
    md = cv2.getTrackbarPos("minDist", win)
    p1 = cv2.getTrackbarPos("param1", win)
    p2 = cv2.getTrackbarPos("param2", win)
    rmin = cv2.getTrackbarPos("minRadius", win)
    rmax = cv2.getTrackbarPos("maxRadius", win)
    return md, p1, p2, rmin, rmax

# ---------------------------------------------------------
# JSONロード／セーブ
# ---------------------------------------------------------
def load_params_if_exist(win, path):
    p = Path(path)
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in data.items():
        try:
            cv2.setTrackbarPos(k, win, int(v))
        except cv2.error:
            # 今のウィンドウに無いパラメータ名は無視
            pass
    print(f"[LOAD] {path}")

def save_params_hsv(path, lo, hi):
    data = {
        "H_low": lo[0],
        "S_low": lo[1],
        "V_low": lo[2],
        "H_high": hi[0],
        "S_high": hi[1],
        "V_high": hi[2],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[SAVE] HSV -> {path}")

def save_params_dis(path, dmin, dmax):
    data = {"Dist_min [cm]": dmin, "Dist_max [cm]": dmax}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[SAVE] Distance -> {path}")

def save_params_hough(path, md, p1, p2, rmin, rmax):
    data = {
        "minDist": md,
        "param1": p1,
        "param2": p2,
        "minRadius": rmin,
        "maxRadius": rmax,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[SAVE] Hough -> {path}")

# ---------------------------------------------------------
# パスのマッピング
# ---------------------------------------------------------
# 追加で使うやつ
HSV_WHITE_PATH = "hsv_params_white.json"   # 8 に該当
HSV_POLL_PATH  = "hsv_params_post.json"    # 10 に該当（新規）

CAM_CONFIG = {
    "d435i": {
        "distance": PARAM_PATH_DIS_D435I,
        "hough": PARAM_HOUGH_D435I,
        "hsv": {
            "ball": PARAM_PATH_HSV[2],   # 既存の青ボール
            "flag": PARAM_PATH_HSV[3],   # 旗
            "white": HSV_WHITE_PATH,     # 白ライン
            "poll": HSV_POLL_PATH,       # ポール
        },
    },
    "d405": {
        "distance": PARAM_PATH_DIS_D405,
        "hough": PARAM_HOUGH_D405,
        "hsv": {
            "ball": PARAM_PATH_HSV[9],   # d405用の青
            "flag": PARAM_PATH_HSV[3],   # 旗は共通でいい
            "white": HSV_WHITE_PATH,
            "poll": HSV_POLL_PATH,
        },
    },
}

def main():
    # ガウシアンの既定値
    try:
        gaus_k, sigmaX = load_filter_params_from_json(PARAM_FILTER)
    except Exception:
        gaus_k, sigmaX = (7, 0)

    # カメラ起動
    W, H, FPS = 640, 480, 15
    cam_d435i = init_realsense_camera(
        name="d435i",
        serial="949122070535",
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
    cam_d405 = init_realsense_camera(
        name="d405",
        serial="218622274519",
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

    time.sleep(1.0)

    # 最初はd435iでボール
    activate_cam = cam_d435i
    active_name = "d435i"
    target = "ball"   # "ball" / "flag" / "white" / "poll"

    # ウィンドウ
    # === Small monitor layout ===
    W_RE, H_RE = 340, 240
    cv2.namedWindow(WIN_INPUT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_HSV_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DIST_CTRL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_HSV_CTRL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_HOUGH_CTRL, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WIN_INPUT, W_RE, H_RE)
    cv2.resizeWindow(WIN_HSV_MASK, W_RE, H_RE)
    cv2.resizeWindow(WIN_RESULT, W_RE, H_RE)
    cv2.resizeWindow(WIN_DIST_CTRL, W_RE, H_RE)
    cv2.resizeWindow(WIN_HSV_CTRL, W_RE, H_RE)
    cv2.resizeWindow(WIN_HOUGH_CTRL, W_RE, H_RE)

    # arrange them on screen so they don't overlap too much
    cv2.moveWindow(WIN_INPUT, 0, 0)
    cv2.moveWindow(WIN_HSV_MASK, 320, 0)
    cv2.moveWindow(WIN_RESULT, 0, 250)
    cv2.moveWindow(WIN_HSV_CTRL, 320, 250)
    cv2.moveWindow(WIN_DIST_CTRL, 0, 500)
    cv2.moveWindow(WIN_HOUGH_CTRL, 320, 500)

    # トラックバー
    create_distance_trackbars(WIN_DIST_CTRL)
    create_hsv_trackbars(WIN_HSV_CTRL)
    create_hough_trackbars(WIN_HOUGH_CTRL)

    # 最初の読み込み
    load_params_if_exist(WIN_DIST_CTRL, CAM_CONFIG[active_name]["distance"])
    load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
    load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])

    print("===== KEY =====")
    print("1: Select D435i")
    print("2: Select D405")
    print("b: Select Ball")
    print("f: Select Flag")
    print("w: Select White  -> hsv_params_white.json")
    print("p: Select Poll   -> hsv_params_post.json")
    print("s: Save current camera + target JSON")
    print("q / ESC: Exit")
    print("================")

    try:
        while True:
            color_image, depth_image = get_rgbd_images(activate_cam)
            if color_image is None:
                continue

            # トラックバー値取得
            dist_min_cm, dist_max_cm = get_distance_range(WIN_DIST_CTRL)
            lo, hi = get_hsv_range(WIN_HSV_CTRL)
            h_minDist, h_p1, h_p2, h_rmin, h_rmax = get_hough_params(WIN_HOUGH_CTRL)

            # 共通前処理
            mask_morph, vis = preprocess_depth_and_hsv(
                color_image,
                depth_image,
                activate_cam,
                dist_min_cm=dist_min_cm,
                dist_max_cm=dist_max_cm,
                gaus_k=gaus_k,
                sigmaX=sigmaX,
                hsv_lo=lo,
                hsv_hi=hi,
            )

            # 上に何をいじってるか表示
            input_show = color_image.copy()
            cv2.putText(
                input_show,
                f"Cam: {active_name}  Target: {target}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # 対象ごとのざっくり表示
            if target == "ball":
                # 一時的にカメラのハフ値を上書きして検出してみる
                old = (activate_cam.minDist,
                       activate_cam.param1,
                       activate_cam.param2,
                       activate_cam.minRadius,
                       activate_cam.maxRadius)
                activate_cam.minDist = h_minDist
                activate_cam.param1 = h_p1
                activate_cam.param2 = h_p2
                activate_cam.minRadius = h_rmin
                activate_cam.maxRadius = h_rmax

                circles = circularity_and_hough(
                    mask_morph,
                    vis,
                    activate_cam,
                    area_min=100,
                    circ_min=0.80,
                )
                found, x, y, r, cam3d, rob3d = evaluate_circle_detection(
                    circles, depth_image, activate_cam
                )
                if found:
                    cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(vis, "BALL", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 戻す
                (activate_cam.minDist,
                 activate_cam.param1,
                 activate_cam.param2,
                 activate_cam.minRadius,
                 activate_cam.maxRadius) = old

            elif target == "flag":
                triangles = detect_triangles(
                    mask_morph,
                    area_min_label=100,
                    area_min=150,
                    epsilon_ratio=0.08,
                )
                recog_res, rob3d, cam3d, goal_xy_pixel = evaluate_triangles_detection(
                    triangles, depth_image, activate_cam
                )
                if triangles:
                    for tri in triangles:
                        cv2.polylines(vis, [tri], True, (0, 255, 255), 2)
                if recog_res and goal_xy_pixel is not None:
                    cv2.circle(vis, goal_xy_pixel, 4, (0, 0, 255), -1)
                    cv2.putText(vis, "FLAG", (goal_xy_pixel[0] + 10, goal_xy_pixel[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                # white / poll のときは検出ロジックは特に書かないで可視化だけ
                cv2.putText(vis, target.upper(), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 表示
            cv2.imshow("Input", input_show)
            cv2.imshow("HSV Mask Morph", mask_morph)
            cv2.imshow("Result", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            # カメラ切替
            elif key == ord("1"):
                activate_cam = cam_d435i
                active_name = "d435i"
                load_params_if_exist(WIN_DIST_CTRL, CAM_CONFIG[active_name]["distance"])
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])
                print("[INFO] switched to d435i")

            elif key == ord("2"):
                activate_cam = cam_d405
                active_name = "d405"
                load_params_if_exist(WIN_DIST_CTRL, CAM_CONFIG[active_name]["distance"])
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])
                print("[INFO] switched to d405")

            # 対象切替
            elif key == ord("b"):
                target = "ball"
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])
                print("[INFO] target = BALL")

            elif key == ord("f"):
                target = "flag"
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                # ハフは旗では基本使わないが一応ロード
                load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])
                print("[INFO] target = FLAG")

            elif key == ord("w"):
                target = "white"
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                # ハフはそのまま
                print("[INFO] target = WHITE")

            elif key == ord("p"):
                target = "poll"
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
                print("[INFO] target = POLL")

            # 保存
            elif key == ord("s"):
                dmin, dmax = get_distance_range(WIN_DIST_CTRL)
                lo, hi = get_hsv_range(WIN_HSV_CTRL)
                md, p1, p2, rmin, rmax = get_hough_params(WIN_HOUGH_CTRL)

                dis_path = CAM_CONFIG[active_name]["distance"]
                hsv_path = CAM_CONFIG[active_name]["hsv"][target]
                hough_path = CAM_CONFIG[active_name]["hough"]

                save_params_dis(dis_path, dmin, dmax)
                save_params_hsv(hsv_path, lo, hi)

                # ハフはボールのときだけ保存しておく
                if target == "ball":
                    save_params_hough(hough_path, md, p1, p2, rmin, rmax)

                print(f"[SAVE DONE] cam={active_name}, target={target}")

    finally:
        if cam_d435i is not None:
            cam_d435i.pipeline.stop()
        if cam_d405 is not None:
            cam_d405.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
