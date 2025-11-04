# -*- coding: utf-8 -*-
# D435iのジャイロセンサのテストコード（0.5秒ごとに出力）

import pyrealsense2 as rs
import cv2
import time
import math

def main():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        pipeline.start(config)

        while True:
            # フレームセットを待機
            frames = pipeline.wait_for_frames()

            # 加速度センサーフレームを取得
            accel_frame = frames[0].as_motion_frame()
            if accel_frame:
                accel_data = accel_frame.get_motion_data()
                print("Accel:", accel_data.x, accel_data.y, accel_data.z)

            # ジャイロスコープフレームも取得する場合
            gyro_frame = frames[1].as_motion_frame()
            if gyro_frame:
                gyro_data = gyro_frame.get_motion_data()
                print("Gyro:", gyro_data.x, gyro_data.y, gyro_data.z)            
            print("-" * 40)
                
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    except Exception as e:
        print("Error initializing RealSense pipeline:", e)

    finally:
        pipeline.stop()

def main_compute():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    pipeline.start(config)

    prev_time = None

    try:
        
        initial_pos = 0
        now_pos = 0
        
        while True:
            frames = pipeline.wait_for_frames()
            accel_frame = frames[0].as_motion_frame()
            if not accel_frame:
                continue

            # タイムスタンプ取得（単位はミリ秒）
            t = accel_frame.get_timestamp()

            if prev_time is not None:
                dt = (t - prev_time) / 1000.0  # 秒に変換
                accel_data = accel_frame.get_motion_data()
                
                # 距離計算
                distance_x = 0.5 * accel_data.x * (dt ** 2)
                distance_y = 0.5 * accel_data.y * (dt ** 2)
                distance_z = 0.5 * accel_data.z * (dt ** 2)
                displacement = (distance_x ** 2 + distance_y ** 2 + distance_z ** 2) ** 0.5
                
                # 表示
                print(f"dt = {dt:.3f} s | ds = ({distance_x:.3f}, {distance_y:.3f}, {distance_z:.3f}) | Displacement = {displacement*1000:.0f} [mm]")
            
            prev_time = t

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

#botu
def main_compute_3():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    pipeline.start(config)

    # ---- 1) 最初に重力を測る ----
    print("Calibrating gravity...")
    gsum = [0.0, 0.0, 0.0]
    N = 60
    for _ in range(N):
        frames = pipeline.wait_for_frames()
        for f in frames:
            if f.is_motion_frame():
                m = f.as_motion_frame()
                if m.get_profile().stream_type() == rs.stream.accel:
                    d = m.get_motion_data()
                    gsum[0] += d.x
                    gsum[1] += d.y
                    gsum[2] += d.z
        time.sleep(0.005)
    gref = [gsum[0]/N, gsum[1]/N, gsum[2]/N]
    print("gravity =", gref)

    # ---- 2) 積分変数 ----
    prev_time = None
    vx = vy = vz = 0.0
    px = py = pz = 0.0

    ACC_STATIC_THRESH = 0.15
    VEL_DAMPING = 0.9

    try:
        while True:
            frames = pipeline.wait_for_frames()
            accel_frame = None
            for f in frames:
                if f.is_motion_frame():
                    m = f.as_motion_frame()
                    if m.get_profile().stream_type() == rs.stream.accel:
                        accel_frame = m
                        break
            if accel_frame is None:
                continue

            ts = accel_frame.get_timestamp()
            if prev_time is None:
                prev_time = ts
                continue

            dt = (ts - prev_time) / 1000.0
            prev_time = ts

            d = accel_frame.get_motion_data()

            # 重力補正
            ax = d.x - gref[0]
            ay = d.y - gref[1]
            az = d.z - gref[2]

            amag = math.sqrt(ax*ax + ay*ay + az*az)

            if amag < ACC_STATIC_THRESH:
                ax = ay = az = 0.0
                vx *= VEL_DAMPING
                vy *= VEL_DAMPING
                vz *= VEL_DAMPING
            else:
                vx += ax * dt
                vy += ay * dt
                vz += az * dt

            px += vx * dtq１
            py += vy * dt
            pz += vz * dt

            disp = math.sqrt(px*px + py*py + pz*pz)

            # ★表示フォーマットを指定通りに修正
            print(
                f"dt={dt:.3f}s "
                f"| p=({px*1000:.1f},{py*1000:.1f},{pz*1000:.1f}) mm | |p|={pz*1000:.1f} mm"
            )

            time.sleep(0.002)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main_compute()
