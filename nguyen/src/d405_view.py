# d405_rgbd_view.py
import cv2, numpy as np, pyrealsense2 as rs

ctx = rs.context()
dev = ctx.query_devices()[0]
print("Device:", dev.get_info(rs.camera_info.name))

# 1) 全プロファイルを列挙して「DepthとColorで一致する解像度・fps」を探す
depth_profiles, color_profiles = [], []
for s in dev.sensors:
    for p in s.get_stream_profiles():
        vp = p.as_video_stream_profile()
        w,h,fps = vp.width(), vp.height(), vp.fps()
        if p.stream_type() == rs.stream.depth:
            depth_profiles.append((w,h,fps,vp.format()))
        if p.stream_type() == rs.stream.color:
            color_profiles.append((w,h,fps,vp.format()))

# 優先候補（1280x720@30 → 640x480@30 の順で探す）
candidates = [(1280,720,30), (640,480,30)]
use = None
for (w,h,fps) in candidates:
    if any((W==w and H==h and F==fps) for (W,H,F,_) in depth_profiles) and \
       any((W==w and H==h and F==fps) for (W,H,F,_) in color_profiles):
        use=(w,h,fps); break
if use is None:
    # どれでもよいので一致セットの最初を使う
    for W,H,F,_ in depth_profiles:
        if any((w==W and h==H and fps==F) for (w,h,fps,_) in color_profiles):
            use=(W,H,F); break
assert use, "No common Depth/Color profile found"
W,H,FPS = use
print(f"Use profile: {W}x{H}@{FPS}")

# 2) パイプライン開始（Depth/Colorを同じ設定で）
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

def to_xyz(u,v,frame_depth, r=2):
    Zs=[]
    for yy in range(max(0,v-r), min(H,v+r+1)):
        for xx in range(max(0,u-r), min(W,u+r+1)):
            z = frame_depth.get_distance(xx, yy)
            if 0.03 < z < 1.0: Zs.append(z)  # D405は近距離想定
    if not Zs: return None
    Z = float(np.median(Zs))
    X,Y,Z = rs.rs2_deproject_pixel_to_point(intr, [float(u),float(v)], Z)
    return X,Y,Z

try:
    while True:
        fs = pipe.wait_for_frames()
        c = fs.get_color_frame(); d = fs.get_depth_frame()
        if not c or not d: continue

        img = np.asarray(c.get_data())
        dep = np.asarray(d.get_data())
        vis = cv2.applyColorMap(cv2.convertScaleAbs(dep, alpha=255/500), cv2.COLORMAP_JET)

        u,v = W//2, H//2
        xyz = to_xyz(u,v,d)
        if xyz:
            X,Y,Z = xyz
            cv2.circle(img,(u,v),6,(0,255,0),-1)
            cv2.putText(img,f"{X:.3f},{Y:.3f},{Z:.3f} m",(u-120,v-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow("D405 Color (Stereo Module)", img)
        cv2.imshow("D405 Depth", vis)
        if cv2.waitKey(1)==27: break
finally:
    pipe.stop(); cv2.destroyAllWindows()
