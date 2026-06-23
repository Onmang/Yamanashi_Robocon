import pyrealsense2 as rs
import numpy as np
import cv2

# ストリームの設定
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

try:
    while True:
        # フレーム待ち
        frames = pipeline.wait_for_frames()

        #IR１
        ir_frame1 = frames.get_infrared_frame(1)
        ir_image1 = np.asanyarray(ir_frame1.get_data())

        #IR2
        ir_frame2 = frames.get_infrared_frame(2)
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        # RGB
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 深度
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        # 2次元データをカラーマップに変換
        ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
        ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        # イメージの結合
        images = np.vstack(( np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap)) ))

        # 表示
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # q キー入力で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    # ストリーミング停止
    pipeline.stop()