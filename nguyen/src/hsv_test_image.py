import cv2
import numpy as np
import os
import json
from pathlib import Path

# 画像パス
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BUNKER_HSV_PATH = "hsv_bunker.json"
GREEN_HSV_PATH = "hsv_green.json"

def _noop(x): pass

def create_hsv_trackbars(win):
    cv2.createTrackbar('H_low',  win, 0,   179, _noop)
    cv2.createTrackbar('H_high', win, 179, 179, _noop)
    cv2.createTrackbar('S_low',  win, 0,   255, _noop)
    cv2.createTrackbar('S_high', win, 255, 255, _noop)
    cv2.createTrackbar('V_low',  win, 0,   255, _noop)
    cv2.createTrackbar('V_high', win, 255, 255, _noop)

def get_hsv_range(win):
    hl = cv2.getTrackbarPos('H_low',  win)
    hh = cv2.getTrackbarPos('H_high', win)
    sl = cv2.getTrackbarPos('S_low',  win)
    sh = cv2.getTrackbarPos('S_high', win)
    vl = cv2.getTrackbarPos('V_low',  win)
    vh = cv2.getTrackbarPos('V_high', win)
    return (hl, sl, vl), (hh, sh, vh)

def save_params(path, lo, hi):
    data = {'H_low':lo[0],'S_low':lo[1],'V_low':lo[2],
            'H_high':hi[0],'S_high':hi[1],'V_high':hi[2]}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Saved HSV params -> {path}")

def load_params_if_exist(win, path):
    p = Path(path)
    if not p.exists(): return
    data = json.loads(p.read_text(encoding='utf-8'))
    for k, v in data.items():
        cv2.setTrackbarPos(k, win, int(v))
    print(f"Loaded HSV params <- {path}")

def main(img_name="", hsv_param=""):

    # image path（スクリプトと同じフォルダ → yolo/datasets/images_val の順に探す）
    candidates = [os.path.join(CURR_PATH, img_name),
                  os.path.join(CURR_PATH, "yolo", "datasets", "images_val", img_name)]
    img_path = next((p for p in candidates if os.path.isfile(p)), None)
    if img_path is None:
        raise FileNotFoundError(f"画像が見つかりません: {candidates}")
    input_image = cv2.imread(img_path)
    if input_image is None:
        raise IOError(f"画像を読み込めません: {img_path}")

    # 画面
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Control', cv2.WINDOW_NORMAL)
    create_hsv_trackbars('HSV Control')
    load_params_if_exist('HSV Control', hsv_param)

    print("操作: s=パラメータ保存, q/ESC=終了")

    try:
        while True:

            bgr = input_image.copy()

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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_name = "bunker_2.jpg"  # ここにテストしたい画像のファイル名を入れる
    main(img_name=image_name, hsv_param=GREEN_HSV_PATH)

