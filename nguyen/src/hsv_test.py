# -*- coding: utf-8 -*-
# RealSense D405 でカラー(H=0-179, S/V=0-255)のHSV二値化＋トラックバー
# 依存関係: pip install opencv-python pyrealsense2

import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path

PARAM_PATH = ["hsv_params_red.json", "hsv_params_yellow.json", "hsv_params_blue.json", "hsv_params_flag.json"] # 保存先パス選択

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

def make_mask_hsv_with_wrap(hsv, lo, hi):
    """Hueが0/179を跨ぐ場合に対応したinRange。lo=(H,S,V), hi=(H,S,V)"""
    hl, sl, vl = lo
    hh, sh, vh = hi
    # S,Vの範囲マスク（共通）
    sv_low  = np.array([0,  sl, vl], dtype=np.uint8)
    sv_high = np.array([179, sh, vh], dtype=np.uint8)

    if hl <= hh:
        # 通常（跨がない）
        rng_low  = np.array([hl, sl, vl], dtype=np.uint8)
        rng_high = np.array([hh, sh, vh], dtype=np.uint8)
        mask = cv2.inRange(hsv, rng_low, rng_high)
    else:
        # 跨ぐ（例：hl=170, hh=10 → [170..179] ∪ [0..10]）
        rng_low1  = np.array([hl,  sl, vl], dtype=np.uint8)
        rng_high1 = np.array([179, sh, vh], dtype=np.uint8)
        rng_low2  = np.array([0,   sl, vl], dtype=np.uint8)
        rng_high2 = np.array([hh,  sh, vh], dtype=np.uint8)
        mask = cv2.inRange(hsv, rng_low1, rng_high1) | cv2.inRange(hsv, rng_low2, rng_high2)

    # 念のためS,Vだけで外れるのを抑える（Hue無視のSVマスクとAND）
    sv_mask = cv2.inRange(hsv, sv_low, sv_high)
    return mask & sv_mask

def draw_hsv_hist_cv(hsv):
    """MatplotlibなしでHSVヒストグラム画像(横600x縦220)を作って返す（軸ラベル付き）"""
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[180],[0,180])  # Hueは0..179
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])

    # 正規化
    hist_h = cv2.normalize(hist_h, None, 0, 180, cv2.NORM_MINMAX).flatten()
    hist_s = cv2.normalize(hist_s, None, 0, 180, cv2.NORM_MINMAX).flatten()
    hist_v = cv2.normalize(hist_v, None, 0, 180, cv2.NORM_MINMAX).flatten()

    W, Himg = 600, 220
    margin_bottom = 20
    img = np.zeros((Himg, W, 3), dtype=np.uint8)

    # 軸スケール用関数
    def plot_poly(hist, color, bins, xscale):
        pts = []
        for i in range(bins):
            x = int(i * xscale)
            y = Himg - margin_bottom - int(hist[i])
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=1)

    plot_poly(hist_h, (0,0,255),   180, W/180)  # H: 赤
    plot_poly(hist_s, (0,255,0),   256, W/256)  # S: 緑
    plot_poly(hist_v, (255,255,255),256, W/256) # V: 白

    # --- 軸描画 ---
    y_base = Himg - margin_bottom
    cv2.line(img, (0, y_base), (W, y_base), (200,200,200), 1)  # x軸
    cv2.line(img, (0, y_base-180), (0, y_base), (200,200,200), 1)  # y軸

    # 横軸目盛（Hue/S/Vの基準）
    tick_step = 30  # Hueの目盛間隔
    for val in range(0, 181, tick_step):
        x = int(val * (W/180))
        cv2.line(img, (x, y_base-5), (x, y_base+3), (150,150,150), 1)
        cv2.putText(img, str(val), (x-10, y_base+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)

    # 縦軸目盛（凡例）
    for i, t in enumerate(["H(0-179)", "S(0-255)", "V(0-255)"]):
        cv2.putText(img, t, (10 + 80*i, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    
                    [(0,0,255), (0,255,0), (255,255,255)][i], 1, cv2.LINE_AA)

    return img

def main(hsv_param_num=0):
    pipeline = rs.pipeline()
    cfg = rs.config()

    # 軽めの解像度（RasPiなら 848x480/30 など推奨）
    W, H, FPS = 848, 480, 30

    # まずカラーを試す（D405はRGB8が出る）。失敗したらIRへフォールバック
    use_color = True
    try:
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    except Exception as e:
        print("Color stream enable failed, fallback to IR:", e)
        use_color = False
        cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)

    profile = pipeline.start(cfg)

    # 画面
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV Control', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask Morph', cv2.WINDOW_NORMAL)
    # ヒストグラムは必要時のみ開く
    show_hist = False

    create_hsv_trackbars('HSV Control')
    load_params_if_exist('HSV Control', PARAM_PATH[hsv_param_num])

    # ウィンドウ配置
    win_w, win_h = 450, 400
    offset_x, offset_y = 50, 50
    cv2.moveWindow('Input', offset_x, offset_y)
    cv2.moveWindow('Mask', win_w + offset_x, offset_y)
    cv2.moveWindow('Mask Morph', 2 * win_w + offset_x, offset_y)
    cv2.moveWindow('Result', offset_x, win_h + offset_y)
    cv2.moveWindow('HSV Control', win_w + offset_x, win_h + offset_y)

    print("操作: s=パラメータ保存, h=ヒスト表示ON/OFF, q/ESC=終了")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            if use_color:
                frame = frames.get_color_frame()
                if not frame:
                    continue
                bgr = np.asanyarray(frame.get_data())
            else:
                ir_frame = frames.get_infrared_frame(1)  # 左IR
                if not ir_frame:
                    continue
                ir = np.asanyarray(ir_frame.get_data())  # (H, W) uint8
                bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

            # 低ノイズ化（任意）
            bgr = cv2.GaussianBlur(bgr, (5,5), 0)

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lo, hi = get_hsv_range('HSV Control')

            # Hueラップ対応マスク
            mask = make_mask_hsv_with_wrap(hsv, lo, hi)

            # 膨張・収縮でマスク整形（任意）
            kernel = np.ones((3,3), np.uint8)
            mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=3)

            # 可視化（マスクをカラーに適用）
            vis = cv2.bitwise_and(bgr, bgr, mask=mask_morph)

            # 輪郭抽出
            contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0,255,0), 2)

            # 面積
            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     if area < 500:  # 小さいノイズは無視
            #         continue
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     cv2.rectangle(vis, (x,y), (x+w, y+h), (255,0,0), 2)
            #     cv2.putText(vis, f"{area:.0f}", (x, y-10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            approx_contours = []
            for i, cnt in enumerate(contours):
                # 輪郭の周囲の長さを計算する。
                arclen = cv2.arcLength(cnt, True)
                # 輪郭を近似する。
                approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.1 * arclen, closed=True)
                approx_contours.append(approx_cnt)
            # 三角形を探す
            triangles = list(filter(lambda x: len(x) == 3, approx_contours))
            cv2.drawContours(vis, triangles, -1, (0,0,255), 2)  # 赤で描画

            # 三角形の面積
            for tri in triangles:
                area = cv2.contourArea(tri)
                if area < 500:
                    continue
                x, y, w, h = cv2.boundingRect(tri)
                cv2.putText(vis, f"Tri:{area:.0f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 三角形の中心座標
            for tri in triangles:
                M = cv2.moments(tri)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(vis, (cx, cy), 5, (0,255,0), -1)  # 緑の点で表示

            cv2.imshow('Input', bgr)
            cv2.imshow('Result', vis)
            cv2.imshow('Mask', mask)
            cv2.imshow('Mask Morph', mask_morph)

            if show_hist:
                hist_img = draw_hsv_hist_cv(hsv)
                cv2.imshow('Hist', hist_img)
            else:
                # 表示OFF時は開いていれば閉じる
                cv2.destroyWindow('Hist')

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('s'):
                save_params(PARAM_PATH[hsv_param_num], lo, hi)
            elif k == ord('h'):
                show_hist = not show_hist

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(hsv_param_num=0)  # 0:赤, 1:緑, 2:青
