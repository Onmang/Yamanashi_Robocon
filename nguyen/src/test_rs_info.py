# test_rs_info.py
import pyrealsense2 as rs
import numpy as np

p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = p.start(c)
dev = prof.get_device()
print("Device :", dev.get_info(rs.camera_info.name))
print("Serial :", dev.get_info(rs.camera_info.serial_number))

# 内部パラメータの取得
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
inst_matrix = np.array([[intr.fx, 0, intr.ppx],
                            [0, intr.fy, intr.ppy],
                            [0, 0, 1]])
print(f"Camera Intrinsics: {intr.width}x{intr.height}")
print("Intrinsic Matrix:")
print("[[fx, 0, ppx],")
print(f" [0, fy, ppy],")
print(f" [0, 0, 1]]")
print(inst_matrix)

p.stop()
