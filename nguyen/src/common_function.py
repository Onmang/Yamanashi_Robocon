# -*- coding: utf-8 -*-
# 共通関数

import numpy as np

class CameraParam:
    """カメラパラメータ格納クラス"""
    def __init__(self):
        self.intr = None  # pyrealsense2.intrinsics オブジェクト
        self.fx = 0
        self.fy = 0
        self.ppx = 0
        self.ppy = 0
        self.depth_scale = 0  # デフォルト値

    def add_ins_param(self, intr):
        self.fx = intr.fx
        self.fy = intr.fy
        self.ppx = intr.ppx
        self.ppy = intr.ppy


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
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])

    # 2. Y軸周りの回転 (Pitch)
    Ry = np.array([[np.cos(ry),  0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    # 3. X軸周りの回転 (Roll)
    Rx = np.array([[1, 0,           0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])

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
    u = int(np.clip(u, 0, W-1))
    v = int(np.clip(v, 0, H-1))

    # 小さなROIの中央値でZを安定化
    r = max(1, roi//2)
    u0, u1 = max(0, u-r), min(W, u+r+1)
    v0, v1 = max(0, v-r), min(H, v+r+1)
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