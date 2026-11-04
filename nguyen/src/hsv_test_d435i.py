import cv2
import numpy as np
import pyrealsense2 as rs

from hsv_test_image import (BUNKER_HSV_PATH, GREEN_HSV_PATH,
                            create_hsv_trackbars, get_hsv_range,
                            save_params, load_params_if_exist)

# カメラ設定
WIDTH, HEIGHT, FPS = 640, 480, 30

def main(hsv_param=""):

    # D435i カラーストリーム開始
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)

    # 画面
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Control', cv2.WINDOW_NORMAL)
    create_hsv_trackbars('HSV Control')
    load_params_if_exist('HSV Control', hsv_param)

    print("操作: s=パラメータ保存, q/ESC=終了")

    try:
        while True:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            bgr = np.asanyarray(color_frame.get_data())

            # 低ノイズ化したいときは有効化
            # bgr = cv2.GaussianBlur(bgr, (5,5), 0)

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lo, hi = get_hsv_range('HSV Control')
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

            # 膨張・収縮でマスク整形（任意）
            # kernel = np.ones((3,3), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # 可視化（マスクをカラーに適用）
            vis = cv2.bitwise_and(bgr, bgr, mask=mask)

            cv2.imshow('Input', bgr)
            cv2.imshow('Mask', vis)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('s'):
                save_params(hsv_param, lo, hi)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(hsv_param=GREEN_HSV_PATH)  # BUNKER_HSV_PATH / GREEN_HSV_PATH を切り替え
