# list_profiles.py
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
assert devs.size() > 0, "RealSenseが見つかりません"
dev = devs[0]
print("Device:", dev.get_info(rs.camera_info.name))
for s in dev.sensors:
    print("  Sensor:", s.get_info(rs.camera_info.name))
    for p in s.get_stream_profiles():
        vp = p.as_video_stream_profile()
        print(f"    {p.stream_name():<10} {vp.format()} {vp.width()}x{vp.height()} @ {vp.fps()}fps")
