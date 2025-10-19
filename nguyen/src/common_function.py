# -*- coding: utf-8 -*-
# 共通関数

import numpy as np

class PositionParam:
    """ロボット位置姿勢格納、マップ情報クラス（グローバル＝ロボット座標系）"""
    def __init__(self):
        self.goal_vector = np.array([[0.0],[0.0]])  # ロボ視点のゴール座標[m]
        self.goal_deg = 0.0                         # （必要なら）ゴール方位[deg]
        self.robot_vector = np.array([[0.0],[0.0]]) # 常に原点（使わない）

    def compute_goal_position(self, dx, dy, move_angle_deg):
        """
        ロボットが自分基準で (dx, dy) 平行移動し、+move_angle_deg 回転したとき、
        ロボ視点（=グローバル）で見えるゴール座標を更新する。
        """
        g = self.goal_vector.copy()
        t = np.array([[dx],[dy]], dtype=float)

        # 座標系が +θ 回転 → 物体は見かけ上 −θ 回転
        theta = np.deg2rad(-move_angle_deg)
        R_neg = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])

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
            tx= -(32.5*0.001), ty=0, tz=110*0.001,
            rx=deg(-90), ry=deg(0), rz=deg(0)
        )


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

    #「y軸を前方向（ロボットの進行方向）」にしたい場合は 引数を入れ替える (atan2(x, y))
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

