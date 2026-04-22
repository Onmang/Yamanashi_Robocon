import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
devices = ctx.query_devices()
assert len(devices) >= 2, "2台のカメラが必要です"

serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
names   = [d.get_info(rs.camera_info.name) for d in devices]

pipes = []
cfgs  = []
for serial in serials:
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfgs.append(cfg)
    pipes.append(rs.pipeline())

profiles = [p.start(c) for p, c in zip(pipes, cfgs)]

try:
    while True:
        frames = [p.wait_for_frames() for p in pipes]
        imgs = []
        for i, f in enumerate(frames):
            color = np.asanyarray(f.get_color_frame().get_data())
            depth = np.asanyarray(f.get_depth_frame().get_data())
            depth_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.02), cv2.COLORMAP_JET)
            combined = np.hstack((color, depth_cmap))
            imgs.append(combined)

        stacked = np.vstack(imgs)
        cv2.imshow("2 Cameras (color+depth)", stacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for p in pipes: p.stop()
    cv2.destroyAllWindows()
