# quick_depth_only.py
import cv2, numpy as np, pyrealsense2 as rs
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)
try:
    while True:
        fs = pipe.wait_for_frames()
        d = fs.get_depth_frame()
        if not d: continue
        dep = np.asanyarray(d.get_data())
        vis = cv2.applyColorMap(cv2.convertScaleAbs(dep, alpha=255/500), cv2.COLORMAP_JET)
        cv2.imshow('Depth', vis)
        if cv2.waitKey(1)==27: break
finally:
    pipe.stop(); cv2.destroyAllWindows()
