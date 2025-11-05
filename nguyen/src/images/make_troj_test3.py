import cv2
import numpy as np

IMG_PATH = "original.png"

WIN_HSV     = "HSV Control"
WIN_MASK    = "Mask / Morph"
WIN_LABEL   = "Labeling All"
WIN_DIST    = "Distance Map"
WIN_RESULT  = "Result"
WIN_ALT     = "Alt Path"          # 角度で探す代替経路
WIN_HSCAN   = "Horizontal Scan"   # ゴールと同じYで横に探す代替経路

# マウス座標を保持するグローバル変数（初期値）
mouse_x, mouse_y = -1, -1
# クリックで決定したゴール座標（未設定は -1）
goal_x, goal_y = -1, -1

def _noop(x): pass

# -------------------------------------------------
# 安全中心をとる関数
# -------------------------------------------------
def get_safe_center(dist: np.ndarray, alpha: float):
    _, maxVal, _, _ = cv2.minMaxLoc(dist)
    safe_mask = dist > alpha * maxVal
    ys, xs = np.where(safe_mask)

    if len(xs) == 0:
        return -1, -1, -1

    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy, dist[cy, cx]

# -------------------------------------------------
# 2点の間を step ピクセルおきにサンプルする
# -------------------------------------------------
def line_sample_points(p0, p1, step=3):
    """
    p0 から p1 までの直線を step ピクセル間隔でサンプリングし、
    各点の (x, y) 座標を順に返すジェネレータ関数。
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    
    # 線分の長さ（ピクセル単位）
    length = int(np.hypot(dx, dy))
    if length == 0:
        yield x0, y0 # 始点と終点が同じ場合はその点のみ返す
        return

    vx = dx / length  # x方向の単位ベクトル
    vy = dy / length  # y方向の単位ベクトル

    for t in range(0, length + 1, step):
        # 現在の位置を整数ピクセルに丸めて返す
        x = int(round(x0 + vx * t))
        y = int(round(y0 + vy * t))
        yield x, y

# -------------------------------------------------
# distanceTransformを使って、直線上に「境界が近い場所」があるか見る
# -------------------------------------------------
def is_path_clear_by_dist(dist_img, p_robot, p_goal,
                          step=3, min_safe_dist=5.0):
    """
    distanceTransform結果(dist_img)を参照し、
    ロボット(p_robot)からゴール(p_goal)までの直線経路上に
    min_safe_dist 未満の領域（＝障害物に近い点）があるかを判定する。
    True = 経路が安全, False = 危険（障害物あり）
    """
    h, w = dist_img.shape[:2]
    # 経路上の点を step ピクセル間隔でサンプリングして調べる
    for x, y in line_sample_points(p_robot, p_goal, step=step):
        if not (0 <= x < w and 0 <= y < h):
            continue    # 画像範囲外はスキップ
        d = dist_img[y, x]  # 障害物までの距離
        if d < min_safe_dist:
            return False    # 安全距離未満の箇所があれば危険
    return True  # すべて安全距離以上 → 経路クリア

# -------------------------------------------------
# ライン上の「最小クリアランス（境界までの最短距離）」を返す
#   → 値が大きいほど安全、0に近いほど危険
# -------------------------------------------------
def line_clearance(dist_img, p_robot, p_goal, step=3):
    """
    distanceTransform結果(dist_img)に基づき、
    ロボット(p_robot)からゴール(p_goal)までの直線経路上で
    最も障害物に近かった距離（＝最小クリアランス）を求める。

    Parameters
    ----------
    dist_img : np.ndarray
        distanceTransform の結果（各画素の障害物までの距離）
    p_robot : tuple[int, int]
        ロボットの画像座標 (x, y)
    p_goal : tuple[int, int]
        ゴールの画像座標 (x, y)
    step : int, optional
        サンプリング間隔（ピクセル単位）

    Returns
    -------
    float
        経路上で最も小さかった距離値（大きいほど安全）
        ※ 範囲外のみの場合は 0.0 を返す
    """
    h, w = dist_img.shape[:2]
    min_d = 1e9  # 初期値（十分大きな数）

    # 経路上を step ピクセル間隔でサンプリング
    for x, y in line_sample_points(p_robot, p_goal, step=step):
        if not (0 <= x < w and 0 <= y < h):
            continue  # 画像外は無視
        d = dist_img[y, x]  # 現在位置の距離値
        if d < min_d:
            min_d = d  # 最小値を更新

    # 1点も有効でなければ 0.0（無効扱い）
    if min_d == 1e9:
        min_d = 0.0

    return min_d


# -------------------------------------------------
# BLOCKED のときに、ロボット中心から放射状に探索して
# 一番遠くまで行ける方向を見つける
# -------------------------------------------------
def find_best_direction(dist_img, origin,
                        angle_step_deg=10,
                        ray_step_px=3,
                        min_safe_dist=5.0):
    h, w = dist_img.shape[:2]
    ox, oy = origin

    best_len = 0
    best_pt = (ox, oy)

    for angle_deg in range(0, 360, angle_step_deg):
        theta = np.deg2rad(angle_deg)
        dx = np.cos(theta)
        dy = np.sin(theta)

        length_px = 0
        while True:
            x = int(round(ox + dx * length_px))
            y = int(round(oy + dy * length_px))

            if not (0 <= x < w and 0 <= y < h):
                break

            d = dist_img[y, x]
            if d < min_safe_dist:
                break

            length_px += ray_step_px

        if length_px > best_len:
            best_len = length_px
            end_x = int(round(ox + dx * (length_px - ray_step_px)))
            end_y = int(round(oy + dy * (length_px - ray_step_px)))
            best_pt = (end_x, end_y)

    return best_pt, best_len

# -------------------------------------------------
# goalと同じYのライン上を、横方向にサンプルして
# 一番安全に行ける点を探す（min_safe_dist対応版）
# -------------------------------------------------
def find_best_horizontal(dist_img, origin, goal_y,
                         x_step=5, step_along_line=3,
                         min_safe_dist=10.0):
    """
    origin         : (x,y) ロボット位置
    goal_y         : ゴールと同じ y（この高さで横に走査する）
    x_step         : 横方向に何ピクセルおきにサンプルするか
    step_along_line: ロボ→候補点 を何ピクセルおきに評価するか
    min_safe_dist  : この距離未満の経路は除外する（安全閾値）
    """
    h, w = dist_img.shape[:2]
    ox, oy = origin

    best_score = -1.0
    best_pt = (ox, goal_y)

    for x in range(0, w, x_step):
        cand = (x, goal_y)
        # このラインの最も狭い場所の距離を評価
        score = line_clearance(dist_img, origin, cand, step=step_along_line)

        # 一定距離未満の経路はスキップ
        if score < min_safe_dist:
            continue

        if score > best_score:
            best_score = score
            best_pt = cand

    return best_pt, best_score

# -------------------------------------------------
# マウスコールバック
# -------------------------------------------------
def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, goal_x, goal_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        goal_x, goal_y = x, y
        mouse_x, mouse_y = x, y

def main():
    global mouse_x, mouse_y, goal_x, goal_y

    src = cv2.imread(IMG_PATH)
    if src is None:
        raise FileNotFoundError(f"{IMG_PATH} が見つかりません")

    # ウィンドウ設定
    cv2.namedWindow(WIN_HSV,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_LABEL,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DIST,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_ALT,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_HSCAN,  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_RESULT, on_mouse)

    # HSVトラックバー
    cv2.createTrackbar("H_low",  WIN_HSV, 26, 179, _noop)
    cv2.createTrackbar("H_high", WIN_HSV, 93, 179, _noop)
    cv2.createTrackbar("S_low",  WIN_HSV, 0,   255, _noop)
    cv2.createTrackbar("S_high", WIN_HSV, 255, 255, _noop)
    cv2.createTrackbar("V_low",  WIN_HSV, 0,   255, _noop)
    cv2.createTrackbar("V_high", WIN_HSV, 255, 255, _noop)

    # alphaトラックバー
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
        
        # 可視化（マスクをカラーに適用）
        vis = cv2.bitwise_and(src, src, mask=mask).copy()

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

        # 代替ルート用の画像を準備
        alt_img   = vis.copy()
        hscan_img = vis.copy()

        # --- 6. 距離変換 + α平均点 + 経路チェック ---
        if np.count_nonzero(mask_largest) > 0:
            h, w = vis.shape[:2]

            dist = cv2.distanceTransform(mask_largest, cv2.DIST_L2, 5)
            alpha_percent = cv2.getTrackbarPos("alpha(%)", WIN_HSV)
            alpha = alpha_percent / 100.0

            cx, cy, radius = get_safe_center(dist, alpha)

            # 距離マップを可視化
            dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
            dist_norm = dist_norm.astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)

            if cx != -1 and cy != -1:
                # セーフセンターの可視化
                radius = int(radius)
                radius = min(radius, int(np.hypot(w, h)))
                cv2.ellipse(vis, (cx, cy), (radius, radius), 0, 0, 180,
                            (0,255,0), 2)
                cv2.circle(vis, (cx, cy), 6, (0,0,255), -1)
                cv2.putText(vis, f"alpha={alpha:.2f}", (cx+5, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # --- ロボット位置（仮） ---
                robot_xy = (w // 2, h - 10)

                # --- ゴール（クリックがなければセーフセンター） ---
                if goal_x >= 0 and goal_y >= 0:
                    goal_xy = (goal_x, goal_y)
                else:
                    goal_xy = (cx, cy)

                # --- 経路チェック ---
                path_clear = is_path_clear_by_dist(
                    dist, robot_xy, goal_xy,
                    step=3,
                    min_safe_dist=10.0
                )

                color = (0,255,0) if path_clear else (0,0,255)
                cv2.line(vis, robot_xy, goal_xy, color, 2)
                cv2.putText(vis,
                            "PATH OK" if path_clear else "BLOCKED",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            color,
                            2)

                # ゴールを分かりやすく
                cv2.circle(vis, goal_xy, 5, (0,255,255), -1)
                cv2.putText(vis, "goal", (goal_xy[0]+5, goal_xy[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                # --- 角度で探す代替ルート（前のやつ） ---
                if not path_clear:
                    best_pt, best_len = find_best_direction(
                        dist, robot_xy,
                        angle_step_deg=10,
                        ray_step_px=3,
                        min_safe_dist=10.0
                    )
                    cv2.line(alt_img, robot_xy, best_pt, (255,0,0), 2)
                    cv2.circle(alt_img, best_pt, 5, (255,0,0), -1)
                    cv2.putText(alt_img, f"best_len={best_len:.1f}",
                                (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                # --- 横に探す代替ルート（今回追加したやつ） ---
                # ゴールが決まっていないときは横探索も意味がないので
                    best_h_pt, best_h_score = find_best_horizontal(
                        dist_img=dist,
                        origin=robot_xy,
                        goal_y=goal_xy[1],
                        x_step=5,
                        step_along_line=3, 
                        min_safe_dist=2.0
                    )
                    cv2.line(hscan_img, robot_xy, best_h_pt, (255,255,0), 2)
                    cv2.circle(hscan_img, best_h_pt, 5, (255,255,0), -1)
                    cv2.putText(hscan_img, f"clear={best_h_score:.1f}",
                                (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255,255,0), 2)
                
                else:
                    cv2.putText(alt_img, "no alt needed",
                                (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(hscan_img, "no alt needed",
                                (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            else:
                dist_color = np.zeros((*mask.shape,3), np.uint8)
                vis = src.copy()
                alt_img = vis.copy()
                hscan_img = vis.copy()
                cv2.putText(vis, "No Safe Area", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            dist_color = np.zeros((*mask.shape,3), np.uint8)
            vis = src.copy()
            alt_img = vis.copy()
            hscan_img = vis.copy()
            cv2.putText(vis, "No target", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # マウス座標の表示
        cv2.putText(vis, f"({mouse_x},{mouse_y})",
                    (10, vis.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1)

        # --- 表示 ---
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_LABEL, label_img)
        cv2.imshow(WIN_DIST, dist_color)
        cv2.imshow(WIN_RESULT, vis)
        cv2.imshow(WIN_ALT, alt_img)
        cv2.imshow(WIN_HSCAN, hscan_img)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
