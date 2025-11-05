import cv2
import numpy as np

IMG_PATH = "original.png"

WIN_HSV     = "HSV Control"
WIN_MASK    = "Mask / Morph"
WIN_LABEL   = "Labeling All"
WIN_DIST    = "Distance Map"
WIN_RESULT  = "Result"

def _noop(x): pass

def main():
    src = cv2.imread(IMG_PATH)
    if src is None:
        raise FileNotFoundError(f"{IMG_PATH} が見つかりません")

    # ウィンドウ設定
    cv2.namedWindow(WIN_HSV,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_LABEL,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DIST,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    # HSVトラックバー
    cv2.createTrackbar("H_low",  WIN_HSV, 26,   179, _noop)
    cv2.createTrackbar("H_high", WIN_HSV, 93, 179, _noop)
    cv2.createTrackbar("S_low",  WIN_HSV, 0,   255, _noop)
    cv2.createTrackbar("S_high", WIN_HSV, 255, 255, _noop)
    cv2.createTrackbar("V_low",  WIN_HSV, 0,   255, _noop)
    cv2.createTrackbar("V_high", WIN_HSV, 255, 255, _noop)

    # alphaトラックバー（安全領域のしきい値 %）
    cv2.createTrackbar("alpha(%)", WIN_HSV, 90, 100, _noop)

    while True:
        # --- 1. ガウシアンぼかし ---
        blur = cv2.GaussianBlur(src, (7,7), 0)

        # --- 2. HSV変換 & 2値化 ---
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h_low  = cv2.getTrackbarPos("H_low",  WIN_HSV)
        h_high = cv2.getTrackbarPos("H_high", WIN_HSV)
        s_low  = cv2.getTrackbarPos("S_low",  WIN_HSV)
        s_high = cv2.getTrackbarPos("S_high", WIN_HSV)
        v_low  = cv2.getTrackbarPos("V_low",  WIN_HSV)
        v_high = cv2.getTrackbarPos("V_high", WIN_HSV)
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # --- 3. モルフォロジ（開→閉） ---
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # --- 4. ラベリング ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        label_hue = np.uint8(179 * labels / np.max(labels)) if np.max(labels) > 0 else labels
        blank_ch = 255 * np.ones_like(label_hue)
        label_img = cv2.merge([label_hue, blank_ch, blank_ch])
        label_img = cv2.cvtColor(label_img, cv2.COLOR_HSV2BGR)
        label_img[label_hue == 0] = 0

        # --- 5. 最大領域のみ抽出 ---
        mask_largest = np.zeros_like(mask)
        if num_labels > 1:
            max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_largest[labels == max_idx] = 255

        # --- 6. 距離変換 + α平均点 ---
        if np.count_nonzero(mask_largest) > 0:
            vis = src.copy()
            dist = cv2.distanceTransform(mask_largest, cv2.DIST_L2, 5)
            alpha_percent = cv2.getTrackbarPos("alpha(%)", WIN_HSV)
            alpha = alpha_percent / 100.0
            
            cx, cy, radius = get_safe_center(dist, alpha)
            h, w = vis.shape[:2]
            radius = int(radius)
            radius = min(radius, int(np.hypot(w, h)))  # 過大防止

            # 半円（上半分）を描く
            cv2.ellipse(
                vis,
                (cx, cy),                   # 中心
                (radius, radius),           # 半径（x, y）
                0,                          # 回転角度
                0, 180,                     # 開始角・終了角 [deg]
                (0, 255, 0),                # 緑色
                2                           # 線の太さ
            )


            # 結果画像に円を描画
            cv2.circle(vis, (cx, cy), radius, (0, 255, 0), 2)  # 緑の円
            # 距離マップを可視化
            dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
            dist_norm = dist_norm.astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
            
            if cx != -1 and cy != -1:
                cv2.circle(vis, (cx, cy), 6, (0,0,255), -1)
                cv2.putText(vis, f"alpha={alpha:.2f}", (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            else:
                cv2.putText(vis, "No Safe Area", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            dist_color = np.zeros((*mask.shape,3), np.uint8)
            vis = src.copy()
            cv2.putText(vis, "No target", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # --- 7. 表示 ---
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_LABEL, label_img)
        cv2.imshow(WIN_DIST, dist_color)
        cv2.imshow(WIN_RESULT, vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cv2.destroyAllWindows()

import numpy as np
import cv2

def get_safe_center(dist: np.ndarray, alpha: float):
    """
    distanceTransform結果 dist に対して、
    境界から十分離れた領域（dist > alpha * maxVal）の重心を求める。

    Parameters
    ----------
    dist : np.ndarray
        cv2.distanceTransform の出力（float32）
    alpha : float
        最大値に対する割合 (例: 0.9 → 最大値の90%以上を安全領域とする)

    Returns
    -------
    (cx, cy) : tuple[int, int]
        安全領域の中心座標。領域がなければ (-1, -1) を返す。
    """
    # 最大値を取得
    _, maxVal, _, _ = cv2.minMaxLoc(dist)

    # 安全領域をマスク化
    safe_mask = dist > alpha * maxVal
    ys, xs = np.where(safe_mask)

    if len(xs) == 0:
        return -1, -1, -1

    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy, dist[cy, cx]# 半径 [px]


if __name__ == "__main__":
    main()
