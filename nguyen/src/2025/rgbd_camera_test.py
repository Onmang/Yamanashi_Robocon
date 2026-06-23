# save as test_rs.py
import cv2, numpy as np, pyrealsense2 as rs

W,H,FPS = 640,480,30
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)

try:
    while True:
        fs = align.process(pipe.wait_for_frames())
        c = fs.get_color_frame(); d = fs.get_depth_frame()
        if not c or not d: continue

        img = np.asarray(c.get_data())
        # 深度を可視化（カラーマップ）
        depth_np = np.asarray(d.get_data())
        depth_vis = cv2.convertScaleAbs(depth_np, alpha=255/4000)  # 4mスケール
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow('color', img)
        cv2.imshow('depth', depth_vis)
        if cv2.waitKey(1)==27: break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
