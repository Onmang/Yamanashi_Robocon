import time
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg  = rs.config()
cfg.disable_all_streams()  # ←重要
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# リトライ付きで開始
for _ in range(3):
    try:
        prof = pipe.start(cfg)
        break
    except rs.error:
        time.sleep(0.2)
else:
    raise RuntimeError("Failed to start pipeline")

# 捨てフレーム
for _ in range(5): pipe.wait_for_frames()

frames = pipe.wait_for_frames()
depth  = frames.get_depth_frame()
color  = frames.get_color_frame()
assert depth and color
print("Depth:", depth.get_width(), "x", depth.get_height(),
      "| Color:", color.get_width(), "x", color.get_height())

pipe.stop()
