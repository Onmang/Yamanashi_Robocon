import pyrealsense2 as rs
import numpy as np

ctx = rs.context()
devices = ctx.query_devices()
print(f"Detected {len(devices)} devices")

for i, dev in enumerate(devices):
    serial = dev.get_info(rs.camera_info.serial_number)
    name   = dev.get_info(rs.camera_info.name)
    print(f"\n=== Camera {i} ===")
    print("Name  :", name)
    print("Serial:", serial)

    # パイプライン構築
    pipe = rs.pipeline(ctx)
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーム開始
    prof = pipe.start(cfg)

    # カラー画像の内部パラメータを取得
    color_prof = prof.get_stream(rs.stream.color)
    intr = color_prof.as_video_stream_profile().get_intrinsics()

    # 行列を作成
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])

    # 整形して出力
    print(f"Camera Intrinsics ({intr.width}x{intr.height}):")
    print("[[fx, 0, ppx],")
    print(" [0, fy, ppy],")
    print(" [0, 0, 1]]")
    print(K)

    pipe.stop()
