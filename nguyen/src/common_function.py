# -*- coding: utf-8 -*-
# 共通関数

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# パラメータ保存用のファイルパス
PARAM_PATH_DIS_D435I = "distance_params.json"
PARAM_PATH_DIS_GREEN = "distance_green_params.json"
PARAM_PATH_DIS_D405 = "distance_d405_params.json"
PARAM_PATH_HSV = [
    "hsv_params_red.json",  # 0
    "hsv_params_yellow.json",  # 1,  d405
    "hsv_params_blue.json",  # 2
    "hsv_params_flag.json",  # 3
    "hsv_params_green.json",  # 4
    "hsv_params_teaground.json",  # 5
    "hsv_params_laf.json",  # 6
    "hsv_params_banker.json",  # 7
    "hsv_params_white.json",  # 8 コース２のグリーンとゴール付近
    "hsv_params_blue_d405.json",  # 9
    "hsv_params_post.json",  # 10
    "hsv_params_yellow_d435i.json",  # 11
]  # 保存先パス選択
PARAM_HOUGH_D435I = "houghcircles_params.json"
PARAM_HOUGH_D405 = "houghcircles_d405_params.json"
PARAM_FILTER = "gaussian_filter_params.json"  # ノイズフィルタGUIの保存先


class CameraParam:
    """カメラパラメータ格納クラス"""

    def __init__(self, T_mtrix):
        self.intr = None  # pyrealsense2.intrinsics オブジェクト
        self.fx = 0
        self.fy = 0
        self.ppx = 0
        self.ppy = 0
        self.depth_scale = 0  # デフォルト値
        self.stereo_baseline = 0  # デフォルト値
        self.dist_min_cm = 0  # フィルター距離下限 [cm]
        self.dist_max_cm = 4000  # フィルター距離上限 [cm]
        self.T_cam2rob = np.eye(4)  # カメラ→ロボット座標変換行列（4x4同次行列）
        self.minDist = 0
        self.param1 = 0
        self.param2 = 0
        self.minRadius = 0
        self.maxRadius = 0
        self.T_cam2rob = T_mtrix

    def add_ins_param(self, intr):
        self.intr = intr
        self.fx = intr.fx
        self.fy = intr.fy
        self.ppx = intr.ppx
        self.ppy = intr.ppy

    def set_transform(self, stereo_baseline):
        self.stereo_baseline = stereo_baseline
        # 同時変換行列, あらかじめロボット座標と決めておく,
        # 現在はカメラを水平にしている
        deg = np.deg2rad  # ← 関数オブジェクトを代入
        # self.T_cam2rob = create_homogeneous_matrix(
        #     # tx=-(32.5 * 0.001), ty=0, tz=110 * 0.001, rx=deg(-90), ry=deg(0), rz=deg(0)  # d435i
        #      tx=0.0, ty=0.0, tz=0.0, rx=deg(-90), ry=deg(0), rz=deg(0)  # d405
        # )


class CameraParam_ver2:
    """1台のRealSenseカメラに対応するパラメータと状態"""

    def __init__(self, name="cam"):
        self.name = name  # "front", "side" みたいな識別用
        self.pipeline = None  # rs.pipeline()
        self.profile = None  # pipeline.start() の戻り
        self.align = None  # rs.align(color) を保持

        # 内部パラメータ
        self.intr = None  # rs.intrinsics
        self.fx = 0.0
        self.fy = 0.0
        self.ppx = 0.0
        self.ppy = 0.0

        # 深度パラメータ
        self.depth_scale = 0.0
        self.stereo_baseline = 0.0  # mm
        self.minDist = 0
        self.param1 = 0
        self.param2 = 0
        self.minRadius = 0
        self.maxRadius = 0
        self.dist_min_cm = 0  # フィルター距離下限 [cm]
        self.dist_max_cm = 300  # フィルター距離上限 [cm]
        self.dist_min_raw = 0  # フィルター距離下限 (深度画像の生値)
        self.dist_max_raw = 0  # フィルター距離上限 (深度画像の生値)

        # hsvフィルタパラメータ
        self.ball_lo = (0, 0, 0)
        self.ball_hi = (0, 0, 0)

        # 外部パラメータ（カメラ→ロボット）
        self.T_cam2rob = np.eye(4)

    def set_intrinsics(self, intr):
        """intr: rs.video_stream_profile().get_intrinsics()"""
        self.intr = intr
        self.fx = intr.fx
        self.fy = intr.fy
        self.ppx = intr.ppx
        self.ppy = intr.ppy

    def set_extrinsics(self, tx, ty, tz, rx_deg, ry_deg, rz_deg):
        """ロボット座標系への外部パラメータを登録 (自分用のハンドアイ結果など)"""
        self.T_cam2rob = create_homogeneous_matrix(
            tx=tx,
            ty=ty,
            tz=tz,
            rx=np.deg2rad(rx_deg),
            ry=np.deg2rad(ry_deg),
            rz=np.deg2rad(rz_deg),
        )

    def print_info(self):
        print(
            f"\n=================== CameraParam [{self.name}] info ==================="
        )
        print(f"[{self.name}] Intrinsics: {self.intr.width}x{self.intr.height}")
        K = np.array(
            [
                [self.fx, 0, self.ppx],
                [0, self.fy, self.ppy],
                [0, 0, 1],
            ]
        )
        print(f"[{self.name}] K =\n{K}")
        print(f"[{self.name}] depth_scale = {self.depth_scale}")
        print(f"[{self.name}] baseline(mm) = {self.stereo_baseline}")
        print(f"[{self.name}] T_cam2rob =\n{self.T_cam2rob}")
        print(
            f"[{self.name}] Filter distance [cm]: {self.dist_min_cm} - {self.dist_max_cm}"
        )
        print(
            f"[{self.name}] Hough params: minDist={self.minDist}, param1={self.param1}, param2={self.param2}, minRadius={self.minRadius}, maxRadius={self.maxRadius}"
        )
        print(f"[{self.name}] Ball HSV lo={self.ball_lo}, hi={self.ball_hi}")
        print("===================================================================\n")


# カメラ初期化のヘルパー関数
def init_realsense_camera(
    name,
    serial=None,
    width=848,
    height=480,
    fps=30,
    extrinsic_guess=None,
    dis_param_path=None,
    hough_param_path=None,
    ball_hsv_param_path=None,
):
    """
    name:    "front", "side" など
    serial:  このカメラのシリアル番号文字列 (Noneなら最初に見つかったやつ)
    extrinsic_guess: dict 例 {
        "tx": -0.0325, "ty": 0.0, "tz": 0.110,
        "rx_deg": -90, "ry_deg": 0, "rz_deg": 0
    }
    """

    cam = CameraParam_ver2(name=name)

    # --- pipeline/config 準備
    pipeline = rs.pipeline()
    cfg = rs.config()

    # シリアル指定（特定の個体を掴みたい場合）
    if serial is not None:
        cfg.enable_device(serial)

    # ストリーム設定
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(cfg)

    # align(color) 用意
    align = rs.align(rs.stream.color)

    # カメラ内部パラメータ取得・保存
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    cam.set_intrinsics(intr)

    # 深度センサ情報 (スケール, ベースライン)
    depth_sensor = profile.get_device().first_depth_sensor()
    cam.depth_scale = depth_sensor.get_depth_scale()
    cam.stereo_baseline = depth_sensor.get_option(rs.option.stereo_baseline)

    # 外部パラメータ (ロボット基準) をセット
    if extrinsic_guess is not None:
        cam.set_extrinsics(
            tx=extrinsic_guess["tx"],
            ty=extrinsic_guess["ty"],
            tz=extrinsic_guess["tz"],
            rx_deg=extrinsic_guess["rx_deg"],
            ry_deg=extrinsic_guess["ry_deg"],
            rz_deg=extrinsic_guess["rz_deg"],
        )

    # オブジェクトに保持
    cam.pipeline = pipeline
    cam.profile = profile
    cam.align = align

    # 距離フィルタのデフォルト値
    cam.dist_min_cm, cam.dist_max_cm = load_filter_distance_from_json(dis_param_path)

    # ハフ変換パラメータ読み込み
    cam.minDist, cam.param1, cam.param2, cam.minRadius, cam.maxRadius = (
        load_hough_params_from_json(hough_param_path)
    )

    # 距離によるフィルタリングの生値計算
    cam.dist_min_raw = int((cam.dist_min_cm / 100.0) / cam.depth_scale)
    cam.dist_max_raw = int((cam.dist_max_cm / 100.0) / cam.depth_scale)

    # hsvフィルタパラメータ読み込み
    cam.ball_lo, cam.ball_hi = load_hsv_from_json(ball_hsv_param_path)

    cam.print_info()
    return cam


def create_homogeneous_matrix(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
    """
    オイラー角（ZYX順）と並進ベクトルから4x4の同次変換行列を作成します。

    Args:
        tx (float): X軸方向の並進 (単位は任意ですが、一貫させること)
        ty (float): Y軸方向の並進
        tz (float): Z軸方向の並進
        rx (float): X軸周りの回転（ロール） (ラジアン)
        ry (float): Y軸周りの回転（ピッチ） (ラジアン)
        rz (float): Z軸周りの回転（ヨー） (ラジアン)

    Returns:
        numpy.ndarray: 4x4の同次変換行列
    """

    # 1. Z軸周りの回転 (Yaw)
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    # 2. Y軸周りの回転 (Pitch)
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    # 3. X軸周りの回転 (Roll)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )

    # 4. 回転行列を結合 (R = Rz * Ry * Rx)
    # ZYXの順で回転を適用します
    R = Rz @ Ry @ Rx

    # 5. 4x4 同次変換行列の作成
    H = np.identity(4)

    # 左上の3x3に回転行列をセット
    H[0:3, 0:3] = R

    # 右端の列に並進ベクトルをセット
    H[0:3, 3] = [tx, ty, tz]

    return H


# 円中心の3D点を求めてロボット座標へ変換して表示するユーティリティ
def project_center_to_robot(u, v, depth_image, depth_scale, intr, T_cam2rob, roi=5):
    H, W = depth_image.shape[:2]
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    # 小さなROIの中央値でZを安定化
    r = max(1, roi // 2)
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    patch = depth_image[v0:v1, u0:u1]
    nz = patch[patch > 0]
    if nz.size == 0:
        return None, None  # 深度無し

    depth_m = np.median(nz) * depth_scale  # [m]

    # 逆投影（ピンホール）
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    X = (u - ppx) / fx * depth_m
    Y = (v - ppy) / fy * depth_m
    Z = depth_m

    P_cam = np.array([X, Y, Z, 1.0], dtype=np.float64)
    P_rob_h = T_cam2rob @ P_cam
    P_rob = P_rob_h[:3] / max(1e-12, P_rob_h[3])

    return (X, Y, Z), P_rob


# 座標から角度を計算する
def compute_angles_from_position(x, y):
    """
    3D座標から方位角と仰角を計算
    本当はカメラの画角てきに90度以上ないけど、マップにゴールの位置を記録するために、90度以上も扱う。

    Args:
        x (float): X座標
        y (float): Y座標

    Returns:
        tuple: 方位角（ラジアン）
    """

    # 「y軸を前方向（ロボットの進行方向）」にしたい場合は 引数を入れ替える (atan2(x, y))
    angle_deg = np.degrees(np.arctan2(x, y))
    return angle_deg


# 角度を符号に合わせて4桁文字列に変換する
def encode_angle(angle_deg: float) -> str:
    """
    角度を4桁文字列に変換する。
    +35 → '1035', -45 → '0045', 0 → '0000'
    """
    sign_flag = 1 if angle_deg >= 0 else 0
    abs_angle = int(abs(angle_deg)) % 1000  # 上限は999°まで
    return f"{sign_flag}{abs_angle:03d}"  # 1桁+3桁=4桁


# 距離を符号に合わせて5桁文字列に変換する
def encode_distance(sign_flag=1, distance_mm=0):
    """
    距離を5桁文字列に変換する。
    +1000 → '01000', -500 → '00500', 0 → '00000'
    """
    dis_send = f"{sign_flag}{distance_mm % 10000:04d}"
    return dis_send


def encode_distance_ver2(distance_mm=0):
    """
    距離を5桁文字列に変換する。
    +1000 → '01000', -500 → '00500', 0 → '00000'
    先頭1桁が符号(1:正, 0:負)、残り4桁が絶対値(mm)
    """
    # 小数が来てもよいようにいったんintにする
    d = int(distance_mm)

    if d < 0:
        sign_flag = 0
        d = -d  # 絶対値にする
    else:
        sign_flag = 1

    # 4桁に収まるように（0〜9999）
    d = d % 10000

    return f"{sign_flag}{d:04d}"


# json から HSV 閾値を読み込む
def load_hsv_from_json(config_path: str) -> tuple:
    """
    JSONファイルからHSVの閾値を読み込み、
    下限(lo)と上限(hi)のタプルを返す

    Args:
        config_path (str): 設定ファイルのパス

    Returns:
        tuple: (lo, hi) のタプル。
               lo = (H_low, S_low, V_low)
               hi = (H_high, S_high, V_high)
    """
    p = Path(config_path)
    default_data = {
        "H_low": 0,
        "S_low": 0,
        "V_low": 0,
        "H_high": 179,
        "S_high": 255,
        "V_high": 255,
    }
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        # ファイルが存在しない場合
        print(f"Error: {config_path} not found. Using default HSV values.")
        data = default_data

    # 辞書からタプルを作成
    lo = (data["H_low"], data["S_low"], data["V_low"])
    hi = (data["H_high"], data["S_high"], data["V_high"])

    # 2つのタプル (lo, hi) を含むタプルを返す
    return lo, hi


# json から フィルター距離を読み込む
def load_filter_distance_from_json(config_path: str) -> int:
    p = Path(config_path)

    d_min_cm = 0
    d_max_cm = 4000

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        d_min_cm = data["Dist_min [cm]"]
        d_max_cm = data["Dist_max [cm]"]

    return d_min_cm, d_max_cm


# json から ハフ変換パラメータを読み込む
def load_hough_params_from_json(config_path: str) -> dict:
    p = Path(config_path)

    minDist, param1, param2, minRadius, maxRadius = 0, 0, 0, 0, 0

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        minDist = data["minDist"]
        param1 = data["param1"]
        param2 = data["param2"]
        minRadius = data["minRadius"]
        maxRadius = data["maxRadius"]

    return minDist, param1, param2, minRadius, maxRadius


# json から ガウシアンノイズフィルタパラメータを読み込む
def load_filter_params_from_json(config_path: str) -> dict:
    p = Path(config_path)

    k, sigmaX = 0, 0

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        k = data["ksize"]
        sigmaX = data["sigmaX"]

    return k, sigmaX


def compute_center_distance(depth_image, depth_scale, W, H, cx, cy, roi_size=10):
    """
    画像中心付近(roi_size x roi_size)の深度の平均値から距離を求める関数

    Args:
        depth_image (ndarray): 深度画像 (uint16など、RealSenseのZ16想定)
        depth_scale (float): RealSenseのdepth_scale [m/1depth_unit]
        W (int): 画像の幅
        H (int): 画像の高さ
        roi_size (int): 中心から取る正方形ROIの一辺ピクセル数

    Returns:
        center_dist_m (float): 中心近傍の平均距離 [m]
        center_dist_mm (float): 中心近傍の平均距離 [mm]
        avg_dist_raw (float): 深度の生値平均 (スケールかける前, depth単位)
    """

    # ROIの範囲（切り出しの安全ガード付き）
    half = roi_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half - 1)
    y2 = int(cy + half - 1)

    # 中心領域の切り出し
    center_roi = depth_image[y1:y2, x1:x2]

    # 深度0(=無効)を除いた平均
    non_zero_values = center_roi[center_roi > 0]
    if non_zero_values.size > 0:
        avg_dist_raw = float(np.mean(non_zero_values))
    else:
        avg_dist_raw = 0.0

    # スケール適用
    center_dist_m = avg_dist_raw * depth_scale
    center_dist_mm = center_dist_m * 1000.0

    return center_dist_m, center_dist_mm, avg_dist_raw, (x1, y1), (x2, y2)


def change_camera(
    activate_cam, cam_d435i, cam_d405, distance_mm, thre_d435i=800, thre_d405=1000
):
    if activate_cam == cam_d435i and distance_mm < thre_d435i:
        activate_cam = cam_d405
        # print("Switched to D405")
    elif activate_cam == cam_d405 and distance_mm > thre_d405:
        activate_cam = cam_d435i
        # print("Switched to D435i")
    return activate_cam


def get_rgbd_images(activate_cam):
    # Get frameset of color and depth
    frames = activate_cam.pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = activate_cam.align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


# --------------------------------------------
# 画像入力→距離マスク＋フィルター、ガウシアン、hsvマスク、モルフォロジー処理
# --------------------------------------------
def preprocess_depth_and_hsv(
    color_image,
    depth_image,
    activate_cam,
    dist_min_cm=None,
    dist_max_cm=None,
    gaus_k=7,
    sigmaX=0,
    hsv_lo=(0, 0, 0),
    hsv_hi=(179, 255, 255),
):
    # カメラの値利用するかどうか
    if dist_min_cm is None or dist_max_cm is None:
        dist_min_raw = activate_cam.dist_min_raw
        dist_max_raw = activate_cam.dist_max_raw
    else:
        dist_min_raw = int((dist_min_cm / 100.0) / activate_cam.depth_scale)
        dist_max_raw = int((dist_max_cm / 100.0) / activate_cam.depth_scale)

    # 指定範囲内のマスクを作成
    mask = cv2.inRange(depth_image, dist_min_raw, dist_max_raw)

    # マスクを適用してフィルタリング
    filtered_image = cv2.bitwise_and(color_image, color_image, mask=mask)

    ## ガウシアンフィルター ##
    filtered_image = cv2.GaussianBlur(filtered_image, (gaus_k, gaus_k), sigmaX)

    ## HSVマスク作成 ##
    hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))

    ## モルフォロジー変換（オープニング＋クロージング）##
    kernel = np.ones((3, 3), np.uint8)
    mask_morph = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3)

    # モルフォロジーマスク適用
    vis = cv2.bitwise_and(filtered_image, filtered_image, mask=mask_morph).copy()

    return mask_morph, vis


# --------------------------------------------
# ラベリング処理→円形度→ハフ変換
# --------------------------------------------
def circularity_and_hough(mask_morph, vis, activate_cam, area_min=100, circ_min=0.80):
    # 初期化
    circles = None

    ## ラベリング処理 ##
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_morph)

    # 円形度良いものだけ抜き出す
    candidate_mask = np.zeros_like(mask_morph)  # ここに有望な領域だけ塗る

    for i in range(1, retval):  # 0は背景なのでスキップ
        x, y, w, h, area = stats[i]
        cx, cy = int(centroids[i][0]), int(centroids[i][1])

        # 面積フィルタ（元のまま）
        if area < area_min:
            continue

        # このラベルだけ取り出すマスクを作る
        blob_mask = np.zeros_like(mask_morph)
        blob_mask[labels == i] = 255

        # このラベル領域内だけで輪郭をとる
        roi = blob_mask[y : y + h, x : x + w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_round_enough = False  # フラグ
        for cnt in contours:
            area_cnt = cv2.contourArea(cnt)
            if area_cnt <= 0:  # 一応
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue  # 一応

            circularity = (4.0 * np.pi * area_cnt) / (peri * peri)

            # 円形度チェック
            if circularity >= circ_min:
                is_round_enough = True

                # 輪郭が1個でも十分丸いなら、そのラベルを候補にする
                break

        # 丸いと判断できたラベル領域だけ candidate_mask に追加
        if is_round_enough:
            candidate_mask[labels == i] = 255

        # 丸いと判断された領域だけ残した画像を作る
        if np.count_nonzero(candidate_mask) > 0:
            # グレースケール変換
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            gray_for_hough = cv2.bitwise_and(gray, gray, mask=candidate_mask)

            circles = cv2.HoughCircles(
                gray_for_hough,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=activate_cam.minDist,
                param1=activate_cam.param1,
                param2=activate_cam.param2,
                minRadius=activate_cam.minRadius,
                maxRadius=activate_cam.maxRadius,
            )

    return circles


# --------------------------------------------
# 円検出結果の評価
# --------------------------------------------
def evaluate_circle_detection(
    circles,
    depth_image,
    activate_cam,
):
    """
    最もロボットに近く、かつ角度が小さい円を選ぶ

    Args:
        circles: cv2.HoughCirclesの出力
        depth_image: 深度画像
        activate_cam: カメラオブジェクト（intr, depth_scale, T_cam2robを持つ）
    Returns:
        (found_flag, x, y, r, cam3d, rob3d)
            found_flag: Trueなら有効な円あり、Falseなら該当なし
    """
    if circles is None or len(circles[0]) == 0:
        return False, None, None, None, None, None

    best_score = float("inf")
    best_result = None

    for c in np.uint16(np.around(circles))[0, :]:
        x, y, r = int(c[0]), int(c[1]), int(c[2])

        # 半径チェック（小ノイズ除外）
        if r < 10:
            continue

        # --- 3D座標計算 ---
        cam3d, rob3d = project_center_to_robot(
            u=x,
            v=y,
            depth_image=depth_image,
            depth_scale=activate_cam.depth_scale,
            intr=activate_cam.intr,
            T_cam2rob=activate_cam.T_cam2rob,
            roi=7,
        )
        if cam3d is None or rob3d is None:
            continue

        Xr, Yr, Zr = rob3d  # [m]
        dist_mm = np.sqrt(Xr**2 + Yr**2) * 1000.0
        if np.isnan(dist_mm) or dist_mm > 2000:
            continue

        angle_deg = abs(
            compute_angles_from_position(Xr, Yr)
        )  # 正面方向に近いほど小さい

        # --- スコア評価 ---
        score = dist_mm + 5.0 * angle_deg  # 重み5.0は調整可

        if score < best_score:
            best_score = score
            best_result = (x, y, r, cam3d, rob3d)

    if best_result is not None:
        x, y, r, cam3d, rob3d = best_result
        return True, x, y, r, cam3d, rob3d
    else:
        return False, None, None, None, None, None


# --------------------------------------------
# 角度・距離のEMA平滑化
# --------------------------------------------
def smooth_angle_distance(rob3d, ema_alpha, prev_angle, prev_dist):
    """
    ロボット座標から角度と距離を計算し、
    過去値(prev_angle, prev_dist)とEMA平滑化して返す。

    Args:
        cam3d (tuple or np.ndarray): カメラ座標 (Xc, Yc, Zc) [m]
        rob3d (tuple or np.ndarray): ロボット座標 (Xr, Yr, Zr) [m]

    """
    Xr, Yr, _ = rob3d

    # 距離[mm]
    dist_mm_raw = round(np.sqrt(Xr**2 + Yr**2) * 1000.0)

    # 角度[deg]（ロボット+Y基準）
    angle_deg_raw = round(compute_angles_from_position(Xr, Yr))

    # === EMA平滑化 ===
    angle_deg = round(ema_alpha * angle_deg_raw + (1 - ema_alpha) * prev_angle)
    dist_mm = round(ema_alpha * dist_mm_raw + (1 - ema_alpha) * prev_dist)

    return angle_deg, dist_mm


# --------------------------------------------
# 角度・距離のEMA平滑化
# --------------------------------------------
def excute_state_LOST_100(miss_count, thres_1=30, thres_2=75, thres_3=75 * 2):
    # まじで見失った場合
    lost_flag = False
    if miss_count <= thres_1:
        mode = 0
        send_dis = 0
        send_angle = 0
    elif miss_count >= thres_2:
        mode = 4
        send_dis = 0
        send_angle = -10
    elif miss_count <= thres_3:
        mode = 4
        send_dis = 0
        send_angle = 20
    else:
        mode = 0
        send_dis = 0
        send_angle = 0
        lost_flag = True
    return mode, send_dis, send_angle, lost_flag


# --------------------------------------------
# ラベリング処理→三角形検出
# --------------------------------------------
def detect_triangles(mask_morph, area_min_label=200, area_min=200, epsilon_ratio=0.08):
    """
    connectedComponentsWithStats() の結果から三角形を検出して返す関数。
    描画は外で行う。
    Returns:
        approx_contours (list): 三角形の輪郭リスト（各要素はN×1×2のnumpy配列）
    """
    # 初期化
    approx_contours = []

    ## ラベリング処理 ##
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph)

    for i in range(1, retval):
        x, y, w, h, area = stats[i]
        if area < area_min_label:
            continue

        blob_mask = np.uint8(labels == i) * 255
        roi = blob_mask[y : y + h, x : x + w]
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


# --------------------------------------------
# 三角形を評価
# --------------------------------------------
def evaluate_triangles_detection(triangles, depth_image, activate_cam):
    if not triangles:
        return False, None, None, None

    # 初期化
    cam3d_res = None
    rob3d_res = None
    recog_res = False
    valid_triangles = []
    valid_cam3d = []
    valid_rob3d = []  # ← rob座標を丸ごと入れる（[Xr, Yr, Zr]）
    valid_dist_mm = []  # ← 水平距離mmをすぐ使えるように入れておく

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

        Xr, Yr, _ = rob3d  # [m]
        # ロボ座標での水平距離[mm]
        dist_rob_mm = np.sqrt(Xr**2 + Yr**2) * 1000.0

        # 範囲フィルタリング
        if 100 < dist_rob_mm < 4000:
            valid_triangles.append(tri)
            valid_rob3d.append(rob3d)
            valid_cam3d.append(cam3d)
            valid_dist_mm.append(dist_rob_mm)

    # --- 最も近い三角形を選択して、角度・距離を計算 ---
    if valid_rob3d:
        # 一番近い水平距離を持つインデックス
        nearest_idx = int(np.argmin(valid_dist_mm))
        nearest_tri = valid_triangles[nearest_idx]

        # 旗のポール分オフセットする
        edge = find_vertical_edge(nearest_tri)
        if edge is not None:
            p1, p2 = edge
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)
            # この1点だけを3Dにする
            cam3d_b, rob3d_b = project_center_to_robot(
                u=mx,
                v=my,
                depth_image=depth_image,
                depth_scale=activate_cam.depth_scale,
                intr=activate_cam.intr,
                T_cam2rob=activate_cam.T_cam2rob,
                roi=7,
            )
            cam3d_res = cam3d_b
            rob3d_res = rob3d_b
            recog_res = True
        else:
            cam3d_res = valid_cam3d[nearest_idx]
            rob3d_res = valid_rob3d[nearest_idx]
            recog_res = False
    return recog_res, rob3d_res, cam3d_res, (mx, my)


# --------------------------------------------
# 旗のポールを探す
# --------------------------------------------
def find_vertical_edge(tri):
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
    return best_edge


# --------------------------------------------
# ゴールポストを探す
# --------------------------------------------
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

    best_area = 0
    for i in range(1, retval):
        x, y, w, h, area = stats[i]
        if area < area_min_label:
            continue

        aspect = h / w if w > 0 else 0
        if aspect < aspect_min or area < area_min_post:
            continue

        # ROI抽出
        blob_mask = np.uint8(labels == i) * 255
        roi = blob_mask[y : y + h, x : x + w]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps_ratio * arclen, True)

        # ROI→全体座標に戻す
        approx[:, 0, 0] += x
        approx[:, 0, 1] += y

        # 重心を計算
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"]) + x
        cy = int(M["m01"] / M["m00"]) + y

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
        break  # 最大面積の縦長1本でOK

    return found, cam3d, rob3d, post_bbox, center_px


# --------------------------------------------
# 経路の安全を確認
# --------------------------------------------
def check_path_safety(mask_morph, depth_image, activate_cam, alpha=0.8):
    ## ラベリング処理 ##
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph)

    # まずは最大ラベルからマスクを作る
    mask_largest = np.zeros_like(mask_morph)
    if retval > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size > 0:
            max_idx = 1 + np.argmax(areas)  # 0は背景なので +1
            mask_largest[labels == max_idx] = 255

        # 距離変換パート
        if np.count_nonzero(mask_largest) > 0:
            # distanceTransform (float32)
            dist = cv2.distanceTransform(mask_largest, cv2.DIST_L2, 5)

            # 安全中心を計算
            cx_safe, cy_safe, _ = get_safe_center(dist, alpha)
            if cx_safe != -1:
                # 3D投影 → ロボ座標
                cam3d, rob3d = project_center_to_robot(
                    u=cx_safe,
                    v=cy_safe,
                    depth_image=depth_image,
                    depth_scale=activate_cam.depth_scale,
                    intr=activate_cam.intr,
                    T_cam2rob=activate_cam.T_cam2rob,
                    roi=7,
                )
                return True, cam3d, rob3d
    return False, None, None


# --------------------------------------------
# 安全中心を計算する
# --------------------------------------------
def get_safe_center(dist: np.ndarray, alpha: float):
    """
    distanceTransform結果 dist に対して、
    境界から十分離れた領域（dist > alpha * maxVal）の重心を求める。

    Parameters
    ----------
    dist : np.ndarray
        cv2.distanceTransform の出力（float32）
    alpha : float
        最大値に対する割合 (例: 0.9 → 最大値の90%以上を安全領域とする)

    Returns
    -------
    (cx, cy) : tuple[int, int]
        安全領域の中心座標。領域がなければ (-1, -1) を返す。
    """
    # 最大値を取得
    _, maxVal, _, _ = cv2.minMaxLoc(dist)

    # 安全領域をマスク化
    safe_mask = dist > alpha * maxVal
    ys, xs = np.where(safe_mask)

    if len(xs) == 0:
        return -1, -1, -1

    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy, dist[cy, cx]  # 半径 [px]


# -------------------------------------------------
# 2点の間を step ピクセルおきにサンプルする
# -------------------------------------------------
def line_sample_points(p0, p1, step=3):
    """
    p0 から p1 までの直線を step ピクセル間隔でサンプリングし、
    各点の (x, y) 座標を順に返すジェネレータ関数。
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    # 線分の長さ（ピクセル単位）
    length = int(np.hypot(dx, dy))
    if length == 0:
        yield x0, y0  # 始点と終点が同じ場合はその点のみ返す
        return

    vx = dx / length  # x方向の単位ベクトル
    vy = dy / length  # y方向の単位ベクトル

    for t in range(0, length + 1, step):
        # 現在の位置を整数ピクセルに丸めて返す
        x = int(round(x0 + vx * t))
        y = int(round(y0 + vy * t))
        yield x, y


# -------------------------------------------------
# distanceTransformを使って、直線上に「境界が近い場所」があるか見る
# -------------------------------------------------
def is_path_clear_by_dist(dist_img, p_robot, p_goal, step=3, min_safe_dist=5.0):
    """
    distanceTransform結果(dist_img)を参照し、
    ロボット(p_robot)からゴール(p_goal)までの直線経路上に
    min_safe_dist 未満の領域（＝障害物に近い点）があるかを判定する。
    True = 経路が安全, False = 危険（障害物あり）
    """
    h, w = dist_img.shape[:2]
    # 経路上の点を step ピクセル間隔でサンプリングして調べる
    for x, y in line_sample_points(p_robot, p_goal, step=step):
        if not (0 <= x < w and 0 <= y < h):
            continue  # 画像範囲外はスキップ
        d = dist_img[y, x]  # 障害物までの距離
        if d < min_safe_dist:
            return False  # 安全距離未満の箇所があれば危険
    return True  # すべて安全距離以上 → 経路クリア


# -------------------------------------------------
# ライン上の「最小クリアランス（境界までの最短距離）」を返す
#   → 値が大きいほど安全、0に近いほど危険
# -------------------------------------------------
def line_clearance(dist_img, p_robot, p_goal, step=3):
    """
    distanceTransform結果(dist_img)に基づき、
    ロボット(p_robot)からゴール(p_goal)までの直線経路上で
    最も障害物に近かった距離（＝最小クリアランス）を求める。

    Parameters
    ----------
    dist_img : np.ndarray
        distanceTransform の結果（各画素の障害物までの距離）
    p_robot : tuple[int, int]
        ロボットの画像座標 (x, y)
    p_goal : tuple[int, int]
        ゴールの画像座標 (x, y)
    step : int, optional
        サンプリング間隔（ピクセル単位）

    Returns
    -------
    float
        経路上で最も小さかった距離値（大きいほど安全）
        ※ 範囲外のみの場合は 0.0 を返す
    """
    h, w = dist_img.shape[:2]
    min_d = 1e9  # 初期値（十分大きな数）

    # 経路上を step ピクセル間隔でサンプリング
    for x, y in line_sample_points(p_robot, p_goal, step=step):
        if not (0 <= x < w and 0 <= y < h):
            continue  # 画像外は無視
        d = dist_img[y, x]  # 現在位置の距離値
        if d < min_d:
            min_d = d  # 最小値を更新

    # 1点も有効でなければ 0.0（無効扱い）
    if min_d == 1e9:
        min_d = 0.0

    return min_d


# -------------------------------------------------
# BLOCKED のときに、ロボット中心から放射状に探索して
# 一番遠くまで行ける方向を見つける
# -------------------------------------------------
def find_best_direction(
    dist_img, origin, angle_step_deg=10, ray_step_px=3, min_safe_dist=5.0
):
    h, w = dist_img.shape[:2]
    ox, oy = origin

    best_len = 0
    best_pt = (ox, oy)

    for angle_deg in range(0, 360, angle_step_deg):
        theta = np.deg2rad(angle_deg)
        dx = np.cos(theta)
        dy = np.sin(theta)

        length_px = 0
        while True:
            x = int(round(ox + dx * length_px))
            y = int(round(oy + dy * length_px))

            if not (0 <= x < w and 0 <= y < h):
                break

            d = dist_img[y, x]
            if d < min_safe_dist:
                break

            length_px += ray_step_px

        if length_px > best_len:
            best_len = length_px
            end_x = int(round(ox + dx * (length_px - ray_step_px)))
            end_y = int(round(oy + dy * (length_px - ray_step_px)))
            best_pt = (end_x, end_y)

    return best_pt, best_len


# -------------------------------------------------
# goalと同じYのライン上を、横方向にサンプルして
# 一番安全に行ける点を探す（min_safe_dist対応版）
# -------------------------------------------------
def find_best_horizontal(
    dist_img, origin, goal_y, x_step=5, step_along_line=3, min_safe_dist=10.0
):
    """
    origin         : (x,y) ロボット位置
    goal_y         : ゴールと同じ y（この高さで横に走査する）
    x_step         : 横方向に何ピクセルおきにサンプルするか
    step_along_line: ロボ→候補点 を何ピクセルおきに評価するか
    min_safe_dist  : この距離未満の経路は除外する（安全閾値）
    """
    h, w = dist_img.shape[:2]
    ox, oy = origin

    best_score = -1.0
    best_pt = (ox, goal_y)

    for x in range(0, w, x_step):
        cand = (x, goal_y)
        # このラインの最も狭い場所の距離を評価
        score = line_clearance(dist_img, origin, cand, step=step_along_line)

        # 一定距離未満の経路はスキップ
        if score < min_safe_dist:
            continue

        if score > best_score:
            best_score = score
            best_pt = cand

    return best_pt, best_score


# -------------------------------------------------
# goalpath確認統合関数
# -------------------------------------------------
def check_goal_path(mask_morph, robot_xy=(0, 0), goal_xy=(0, 0), alpha=0.8):
    ## ラベリング処理 ##
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morph)

    # まずは最大ラベルからマスクを作る
    mask_largest = np.zeros_like(mask_morph)
    if retval > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size > 0:
            max_idx = 1 + np.argmax(areas)  # 0は背景なので +1
            mask_largest[labels == max_idx] = 255

        # 距離変換パート
        if np.count_nonzero(mask_largest) > 0:
            # distanceTransform (float32)
            dist = cv2.distanceTransform(mask_largest, cv2.DIST_L2, 5)

            # 安全中心を計算
            cx, cy, _ = get_safe_center(dist, alpha)
            if cx != -1 and cy != -1:
                # --- 経路チェック ---
                path_clear = is_path_clear_by_dist(
                    dist, robot_xy, goal_xy, step=3, min_safe_dist=10.0
                )
            return path_clear, dist
    return False, None
