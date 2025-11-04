import cv2
import numpy as np

# 1. 画像の読み込みと2値化
img_path = 'hsv_img.png'
# グレースケール画像として読み込む
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print(f"エラー: 画像ファイル '{img_path}' を読み込めませんでした。")
    exit()

# 念のため、再度2値化して 0 と 255 の値に統一します
# 閾値1で、1以上の値を255にする
_, binary_img = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

print(f"画像を読み込みました: {img_path}")

# ----------------------------------------------------
# 1. ラベリング処理
# ----------------------------------------------------
# 連結しているピクセル領域にラベルを割り振る
# connectivity=8 : 8近傍（上下左右＋斜め）で連結を判断
# num_labels: ラベルの総数 (背景0を含む)
# labels_img: 各ピクセルにラベルIDが振られた画像
# stats: 各ラベルの統計情報 [左端x, 上端y, 幅, 高さ, 面積]
# centroids: 各ラベルの重心 [x, y]
num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(binary_img, 8, cv2.CV_32S)

print(f"ラベリング完了。背景を含む {num_labels} 個の領域を検出。")

# ----------------------------------------------------
# 2. 最大面積の領域を抽出
# ----------------------------------------------------
# ラベル0は背景なので、それ以外の領域（ラベル1から）を対象にする
if num_labels <= 1:
    print("背景以外の領域が見つかりませんでした。")
    exit()

# stats[ラベルID, cv2.CC_STAT_AREA] で面積を取得
# ラベル1以降の面積リストを取得
areas = stats[1:, cv2.CC_STAT_AREA]

# 最大面積のインデックスを取得 (areasのインデックスは0から始まる)
max_label_index = np.argmax(areas)

# 実際のラベルID (areasのインデックス + 1)
max_label_id = max_label_index + 1

max_area = stats[max_label_id, cv2.CC_STAT_AREA]
print(f"最大面積の領域は ラベルID: {max_label_id}, 面積: {max_area} ピクセル です。")

# 最大面積の領域だけを白 (255) にしたマスク画像を作成
max_area_mask = np.zeros(labels_img.shape, dtype=np.uint8)
max_area_mask[labels_img == max_label_id] = 255

# ----------------------------------------------------
# 3. 輪郭を描画する
# ----------------------------------------------------
# 最大面積のマスク画像から輪郭を検出
# cv2.RETR_EXTERNAL: 最も外側の輪郭のみを検出
# cv2.CHAIN_APPROX_SIMPLE: 輪郭の直線部分を圧縮
contours, _ = cv2.findContours(max_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 元の画像をBGR（カラー）に変換して、色付きの輪郭を描画できるようにする
output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# 輪郭を描画
# contours: 検出された輪郭のリスト
# -1: すべての輪郭を描画 (この場合は1つのはず)
# (0, 255, 0): 輪郭の色 (BGR形式で緑)
# 2: 輪郭の太さ
cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

print("最大領域の輪郭を描画しました。")

# ----------------------------------------------------
# 4. 結果の表示
# ----------------------------------------------------
cv2.namedWindow('Original Binary Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour Drawn', cv2.WINDOW_NORMAL)
# 元の画像
cv2.imshow('Original Binary Image', binary_img)
# 輪郭を描画した画像
cv2.imshow('Contour Drawn', output_img)

print("\n結果ウィンドウが表示されています。いずれかのキーを押すと終了します。")
# キー入力を待つ
cv2.waitKey(0)
# すべてのウィンドウを閉じる
cv2.destroyAllWindows()

# (オプション) 結果をファイルに保存する場合
# cv2.imwrite('largest_area_mask.png', max_area_mask)
# cv2.imwrite('contour_output.png', output_img)