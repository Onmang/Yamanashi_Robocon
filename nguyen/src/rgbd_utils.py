# -*- coding: utf-8 -*-
"""rgbd_utils.py

rgbd camera scripts

Author:
    nguyen

Date:
    2026-07-04

History:
    - 2026-07-04: nguyen coped from 2025
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

class CameraParam:
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

        self.dist_min_cm = 0  # フィルター距離下限 [cm]
        self.dist_max_cm = 300  # フィルター距離上限 [cm]
        self.dist_min_raw = 0  # フィルター距離下限 (深度画像の生値)
        self.dist_max_raw = 0  # フィルター距離上限 (深度画像の生値)

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
        print(f"[Info] Intrinsics: {self.intr.width}x{self.intr.height}")
        K = np.array(
            [
                [self.fx, 0, self.ppx],
                [0, self.fy, self.ppy],
                [0, 0, 1],
            ]
        )
        print(f"[Info] K =\n{K}")
        print(f"[Info] depth_scale = {self.depth_scale}")
        print(f"[Info] baseline(mm) = {self.stereo_baseline}")
        print(f"[Info] T_cam2rob =\n{self.T_cam2rob}")
        print(
            f"[Info] Filter distance [cm]: {self.dist_min_cm} - {self.dist_max_cm}"
        )
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

):
    """
    name:    "front", "side" など
    serial:  このカメラのシリアル番号文字列 (Noneなら最初に見つかったやつ)
    extrinsic_guess: dict 例 {
        "tx": -0.0325, "ty": 0.0, "tz": 0.110,
        "rx_deg": -90, "ry_deg": 0, "rz_deg": 0
    }
    """

    cam = CameraParam(name=name)

    # --- pipeline/config 準備
    pipeline = rs.pipeline()
    cfg = rs.config()

    # シリアル指定（特定のcameraを掴みたい場合）
    if serial is not None:
        cfg.enable_device(serial)

    # ストリーム設定
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(cfg)

    # align(color) 用意
    align = rs.align(rs.stream.color)

    # 自動露光が安定するまで数フレーム捨てる
    for _ in range(5):
        pipeline.wait_for_frames()

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

    # 距離によるフィルタリングの生値計算
    cam.dist_min_raw = int((cam.dist_min_cm / 100.0) / cam.depth_scale)
    cam.dist_max_raw = int((cam.dist_max_cm / 100.0) / cam.depth_scale)

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


# json から フィルター距離を読み込む
def load_filter_distance_from_json(config_path: str):
    p = Path(config_path)

    d_min_cm = 0
    d_max_cm = 400

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        d_min_cm = data["Dist_min [cm]"]
        d_max_cm = data["Dist_max [cm]"]

    return d_min_cm, d_max_cm

# get rgb-d
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


# get frame
def get_rgbd_frames(activate_cam):
    # Get frameset of color and depth
    frames = activate_cam.pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = activate_cam.align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    return color_frame, depth_frame



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



# -----------------------------------------------------------------------------
# [get_depth_at_bbox] バウンディングボックス中心の深度 [mm] を取得する
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def get_depth_at_bbox(depth_frame, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    バウンディングボックス中心の深度 [mm] を取得する
    Args:
        depth_frame: 深度フレーム
        x1 (int): バウンディングボックスの左上X座標
        y1 (int): バウンディングボックスの左上Y座標
        x2 (int): バウンディングボックスの右下X座標
        y2 (int): バウンディングボックスの右下Y座標
    Returns:
        float: 深度 [mm]
    """
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth_value = depth_frame.get_distance(cx, cy)  # メートル単位
    return depth_value * 1000  # mm に変換

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
