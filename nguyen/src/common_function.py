# -*- coding: utf-8 -*-
# 共通関数

import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

# パラメータ保存用のファイルパス
PARAM_PATH_DIS = "distance_params.json"
PARAM_PATH_DIS_GREEN = "distance_green_params.json"
PARAM_PATH_DIS_D405 = "distance_d405_params.json"
PARAM_PATH_HSV = [
    "hsv_params_red.json", #0
    "hsv_params_yellow.json", #1
    "hsv_params_blue.json", #2
    "hsv_params_flag.json", #3
    "hsv_params_green.json", #4
    "hsv_params_teaground.json", #5
    "hsv_params_laf.json", #6
    "hsv_params_banker.json", #7
    "hsv_params_white.json",  #8 コース２のグリーンとゴール付近
    "hsv_params_blue_d405.json", #9
]  # 保存先パス選択
PARAM_HOUGH = "houghcircles_params.json"
PARAM_HOUGH_D405 = "houghcircles_d405_params.json"
PARAM_FILTER = "gaussian_filter_params.json"  # ノイズフィルタGUIの保存先


class PositionParam:
    """ロボット位置姿勢格納、マップ情報クラス（グローバル＝ロボット座標系）"""

    def __init__(self):
        self.goal_vector = np.array([[0.0], [0.0]])  # ロボ視点のゴール座標[m]
        self.goal_deg = 0.0  # （必要なら）ゴール方位[deg]
        self.robot_vector = np.array([[0.0], [0.0]])  # 常に原点（使わない）

    def compute_goal_position(self, dx, dy, move_angle_deg):
        """
        ロボットが自分基準で (dx, dy) 平行移動し、+move_angle_deg 回転したとき、
        ロボ視点（=グローバル）で見えるゴール座標を更新する。
        """
        g = self.goal_vector.copy()
        t = np.array([[dx], [dy]], dtype=float)

        # 座標系が +θ 回転 → 物体は見かけ上 −θ 回転
        theta = np.deg2rad(-move_angle_deg)
        R_neg = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        # 先に平行移動を引き、その後に −θ で回転
        #   g_{t+1} = R(-θ) @ ( g_t - t )
        self.goal_vector = R_neg @ (g - t)

        # 角度も座標系回転の逆で更新したいなら（任意）
        self.goal_deg = (self.goal_deg - move_angle_deg) % 360.0


class CameraParam:
    """カメラパラメータ格納クラス"""

    def __init__(self):
        self.intr = None  # pyrealsense2.intrinsics オブジェクト
        self.fx = 0
        self.fy = 0
        self.ppx = 0
        self.ppy = 0
        self.depth_scale = 0  # デフォルト値
        self.stereo_baseline = 0  # デフォルト値
        self.T_cam2rob = np.eye(4)  # カメラ→ロボット座標変換行列（4x4同次行列）

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
        self.T_cam2rob = create_homogeneous_matrix(
            tx=-(32.5 * 0.001), ty=0, tz=110 * 0.001, rx=deg(-90), ry=deg(0), rz=deg(0)
        )


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


# カメラ初期化のヘルパー関数
def init_realsense_camera(
    name, serial=None, width=848, height=480, fps=30, extrinsic_guess=None
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

def change_camera(activate_cam, cam_d435i, cam_d405, distance_mm, thre_d435i=800, thre_d405=1000):
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
    