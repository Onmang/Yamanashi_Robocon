# -*- coding: utf-8 -*-
# recognition_ver2.py
# 会場で距離・HSV・ハフのパラメータを調整するツール
# 2カメラ(D435i / D405)を起動し、キー操作でどちらのカメラの設定をいじるか切り替える
# 2025/11/06 + white/poll 追加版 + poll長方形検出

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
    # ポストを3Dに投影するために使う
    project_center_to_robot,
)

# window names
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
# ポスト検出（←今回追加するやつ）
# ---------------------------------------------------------
def detect_goal_post(
    mask_morph,
    depth_image,
    activate_cam,
    area_min_label=300,
    area_min_post=500,
    aspect_min=3.0,
    approx_eps_ratio=0.03,
):
    """
    白ポストを検出してロボ座標を返す。
    Returns:
        found (bool)
        cam3d (np.ndarray or None)
        rob3d (np.ndarray or None)
        post_bbox (tuple or None): (x, y, w, h)
        center_px (tuple or None): (cx, cy)
    """
    found = False
    cam3d = rob3d = None
    post_bbox = None
    center_px = None

    # ラベリング
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph)

    for i in range(1, retval):
        x, y, w, h, area = stats[i]
        if area < area_min_label:
            continue

        # 縦長チェック
        aspect = (h / w) if w > 0 else 0
        if aspect < aspect_min or area < area_min_post:
            continue

        # このラベルだけ取り出して輪郭を見る
        blob_mask = np.uint8(labels == i) * 255
        roi = blob_mask[y:y + h, x:x + w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps_ratio * arclen, True)

        # ROI→全体座標に戻す
        approx[:, 0, 0] += x
        approx[:, 0, 1] += y

        # 重心（ROI基準 → 全体へ）
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"]) + x
        cy = int(M["m01"] / M["m00"]) + y

        # 3D投影
        cam3d, rob3d = project_center_to_robot(
            u=cx,
            v=cy,
            depth_image=depth_image,
            depth_scale=activate_cam.depth_scale,
            intr=activate_cam.intr,
            T_cam2rob=activate_cam.T_cam2rob,
            roi=7,
        )

        found = True
        post_bbox = (x, y, w, h)
        center_px = (cx, cy)
        break  # 一番条件に合ったやつ1本でOK

    return found, cam3d, rob3d, post_bbox, center_px


# ---------------------------------------------------------
# パスのマッピング
# ---------------------------------------------------------
HSV_WHITE_PATH = "hsv_params_white.json"   # 8
HSV_POLL_PATH = "hsv_params_post.json"     # 10

CAM_CONFIG = {
    "d435i": {
        "distance": PARAM_PATH_DIS_D435I,
        "hough": PARAM_HOUGH_D435I,
        "hsv": {
            "ball": PARAM_PATH_HSV[2],
            "flag": PARAM_PATH_HSV[3],
            "white": HSV_WHITE_PATH,
            "poll": HSV_POLL_PATH,
        },
    },
    "d405": {
        "distance": PARAM_PATH_DIS_D405,
        "hough": PARAM_HOUGH_D405,
        "hsv": {
            "ball": PARAM_PATH_HSV[9],
            "flag": PARAM_PATH_HSV[3],
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
    target = "ball"

    # ウィンドウ（5インチ想定の小さめ）
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

            # 上に今の状態を表示
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

            # 対象ごとの可視化
            if target == "ball":
                old = (
                    activate_cam.minDist,
                    activate_cam.param1,
                    activate_cam.param2,
                    activate_cam.minRadius,
                    activate_cam.maxRadius,
                )
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
                    cv2.putText(
                        vis,
                        "BALL",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                (
                    activate_cam.minDist,
                    activate_cam.param1,
                    activate_cam.param2,
                    activate_cam.minRadius,
                    activate_cam.maxRadius,
                ) = old

            elif target == "flag":
                triangles = detect_triangles(
                    mask_morph,
                    area_min_label=100,
                    area_min=120,
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
                    cv2.putText(
                        vis,
                        "FLAG",
                        (goal_xy_pixel[0] + 10, goal_xy_pixel[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            elif target == "poll":
                # ここでさっき追加したポスト検出を使う
                found, cam3d, rob3d, bbox, center_px = detect_goal_post(
                    mask_morph,
                    depth_image,
                    activate_cam,
                    # 必要ならここでパラメータ調整
                    area_min_label=200,
                    area_min_post=400,
                    aspect_min=3.0,
                )
                cv2.putText(
                    vis,
                    "POLL",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                if found:
                    x, y, w, h = bbox
                    cx, cy = center_px
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        f"({cx},{cy})",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    # デバッグで3D表示したければここで出す
                    if cam3d is not None:
                        cv2.putText(
                            vis,
                            f"Z:{cam3d[2]*1000:.0f}mm",
                            (x, y - 10 if y > 20 else y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

            else:  # white
                cv2.putText(
                    vis,
                    "WHITE",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # 表示
            cv2.imshow(WIN_INPUT, input_show)
            cv2.imshow(WIN_HSV_MASK, mask_morph)
            cv2.imshow(WIN_RESULT, vis)

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
                load_params_if_exist(WIN_HOUGH_CTRL, CAM_CONFIG[active_name]["hough"])
                print("[INFO] target = FLAG")

            elif key == ord("w"):
                target = "white"
                load_params_if_exist(WIN_HSV_CTRL, CAM_CONFIG[active_name]["hsv"][target])
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
