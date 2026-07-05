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

