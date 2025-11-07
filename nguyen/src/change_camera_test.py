# coding: utf-8


import cv2
import time
import numpy as np
import pyrealsense2 as rs

from common_function import (
    compute_angles_from_position,
    encode_angle,
    encode_distance,
    init_realsense_camera,
    load_filter_distance_from_json,
    load_filter_params_from_json,
    load_hough_params_from_json,
    load_hsv_from_json,
    project_center_to_robot,
    compute_center_distance,
    PARAM_PATH_DIS_D435I,
    PARAM_PATH_HSV,
    PARAM_PATH_DIS_D405,
    PARAM_PATH_DIS_GREEN,
    PARAM_HOUGH_D405,
    PARAM_HOUGH_D435I,
    PARAM_FILTER,
)

CHANGE_CAMERA_THRE_D435I = 350 # mm
CHANGE_CAMERA_THRE_D405 = 550 # mm

def main():
    # --------------------------------------------
    # カメラ初期化
    # --------------------------------------------
    # 解像度とFPS
    W, H, FPS = 640, 480, 15
    try:
        # RealSense D435i カメラ初期化
        cam_d435i = init_realsense_camera(
            name="d435i",
            serial="949122070535",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": -(32.5 * 0.001),
                "ty": -50 * 0.001,
                "tz": 200 * 0.001,
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
            dis_param_path=PARAM_PATH_DIS_D435I,
            hough_param_path=PARAM_HOUGH_D435I,
            ball_hsv_param_path=PARAM_PATH_HSV[2],
        )

        # RealSense D405 カメラ初期化
        # d405はcam3d
        cam_d405 = init_realsense_camera(
            name="d405",
            serial="218622274519",  # 実機のシリアル
            width=W,
            height=H,
            fps=FPS,
            extrinsic_guess={
                "tx": 0.0,
                "ty": 0.0,
                "tz": 0.0,
                "rx_deg": -90,
                "ry_deg": 0,
                "rz_deg": 0,
            },
            dis_param_path=PARAM_PATH_DIS_D405,
            hough_param_path=PARAM_HOUGH_D405,
            ball_hsv_param_path=PARAM_PATH_HSV[9],
        )
        
    except Exception as e:
        print("Camera initialization failed:", e)
        return
    finally:
        time.sleep(2)  # カメラ安定化待ち

    # window setup
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Input", 50, 60)

    # init activate cam
    activate_cam = cam_d405

    # main loop
    while True:
        # get rgbd images
        color_image, depth_image = get_rgbd_images(activate_cam)
        
        # 中心座標
        cx, cy = W // 2, H // 2
        center_dist_m, center_dist_mm, _, (x1, y1), (x2, y2) = compute_center_distance(
                    depth_image,
                    activate_cam.depth_scale,  # ← d435iでもd405でもOK
                    W,
                    H,
                    cx,
                    cy,
                    roi_size=20,
                )
        # オーバーレイ画像作成
        dist_text = f"Center Distance: {center_dist_m:.3f} [m] ({center_dist_m * 1000:.0f} [mm])"

        overlay = color_image.copy()
        cv2.putText(
            overlay,
            dist_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # 中心領域を矩形で表示
        cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color=(0, 200, 0),  # 緑枠
                thickness=1,
            )

        cv2.imshow("Input", overlay)
        
        # change camera based on distance
        activate_cam = change_camera(activate_cam, cam_d435i, cam_d405, center_dist_mm)

        key = cv2.waitKey(1)
        if key == ord("1"):
            activate_cam = cam_d435i
            print("Activated camera: D435i")
        elif key == ord("2"):
            activate_cam = cam_d405
            print("Activated camera: D405")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    cam_d435i.pipeline.stop()
    cam_d405.pipeline.stop()


def get_rgbd_images(activate_cam):
        # Get frameset of color and depth
        frames = activate_cam.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = activate_cam.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
def change_camera(activate_cam, cam_d435i, cam_d405, distance_mm):
    if activate_cam == cam_d435i and distance_mm < CHANGE_CAMERA_THRE_D435I:
        activate_cam = cam_d405
        print("Switched to D405")
    elif activate_cam == cam_d405 and distance_mm > CHANGE_CAMERA_THRE_D405:
        activate_cam = cam_d435i
        print("Switched to D435i")
    return activate_cam



if __name__ == '__main__':
    main()