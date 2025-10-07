import cv2
import numpy as np
import os
import ctypes
import sys


def nothing(x):
    pass
    
# コマンドライン引数から画像パスを取得
if len(sys.argv) < 2:
    print("画像ファイルを指定してください（例: python hsv_test_code.py images_data/DSC_0197.JPG）")
    exit()

# どちらの処理を使うか切り替え（True: HSV, False: RGB）
MODE = True

# 画像パス
curr_path = os.getcwd()
img_name = sys.argv[1]
img_path = os.path.join(curr_path, "images_data", img_name)
input_image = cv2.imread(img_path)
if input_image is None:
    print("画像読み込み失敗：", img_path)
    exit()

# ディスプレイ幅取得（Windows）
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
max_display_width = int(screen_width * 0.95)

# リサイズ処理
base_resize_width = 800
h, w = input_image.shape[:2]
resize_ratio = base_resize_width / w
resized_image = cv2.resize(input_image, (base_resize_width, int(h * resize_ratio)))

# トラックバー作成（H/S/V でも R/G/B でも共通流用）
cv2.namedWindow('Control', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control', 400, 300)
cv2.createTrackbar('C1_low', 'Control', 0, 255, nothing)
cv2.createTrackbar('C1_high', 'Control', 255, 255, nothing)
cv2.createTrackbar('C2_low', 'Control', 221, 255, nothing)
cv2.createTrackbar('C2_high', 'Control', 255, 255, nothing)
cv2.createTrackbar('C3_low', 'Control', 161, 255, nothing)
cv2.createTrackbar('C3_high', 'Control', 255, 255, nothing)

while True:
    # スライダーから値取得してソート（low <= high）
    c1_low, c1_high = sorted([cv2.getTrackbarPos('C1_low', 'Control'), cv2.getTrackbarPos('C1_high', 'Control')])
    c2_low, c2_high = sorted([cv2.getTrackbarPos('C2_low', 'Control'), cv2.getTrackbarPos('C2_high', 'Control')])
    c3_low, c3_high = sorted([cv2.getTrackbarPos('C3_low', 'Control'), cv2.getTrackbarPos('C3_high', 'Control')])

    if MODE:
        # HSV変換してマスク作成
        hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        lower = np.array([c1_low, c2_low, c3_low])
        upper = np.array([c1_high, c2_high, c3_high])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # RGBマスク（OpenCVはBGRなので順番注意）
        lower = np.array([c3_low, c2_low, c1_low])  # BGR順
        upper = np.array([c3_high, c2_high, c1_high])
        mask = cv2.inRange(resized_image, lower, upper)

    # マスク適用
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # スケーリング（画面に収める）
    h_disp, w_disp = resized_image.shape[:2]
    combined_width = w_disp * 3
    scale = min(1.0, max_display_width / combined_width)

    img1 = cv2.resize(resized_image, None, fx=scale, fy=scale)
    img2 = cv2.resize(masked_image, None, fx=scale, fy=scale)
    img3 = cv2.resize(mask_bgr, None, fx=scale, fy=scale)

    combined = np.hstack((img1, img2, img3))
    cv2.imshow('HSV/RGB Filter [Input | Masked | Binary]', combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
