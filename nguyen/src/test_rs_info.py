# test_rs_info.py
import pyrealsense2 as rs
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = p.start(c)
dev = prof.get_device()
print("Device :", dev.get_info(rs.camera_info.name))
print("Serial :", dev.get_info(rs.camera_info.serial_number))
p.stop()
