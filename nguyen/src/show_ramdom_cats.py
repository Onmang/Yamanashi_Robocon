import cv2
import glob
import random
import time
import os

# --- 設定 ---
FOLDER_PATH = './neko_img'  # ★ 画像ファイルがあるフォルダのパスを指定
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp'] # 取得する画像の拡張子
WINDOW_NAME = 'Random Image Viewer'
DISPLAY_TIME_MS = 2000  # 表示時間（ミリ秒、2000ms = 2秒）
# --- ---

def get_image_list(folder_path, extensions):
    """指定フォルダ内の画像ファイルのパス一覧を取得する"""
    all_images = []
    for ext in extensions:
        # glob.glob()で指定されたパターンにマッチするファイルパスを取得
        # os.path.join()でフォルダパスと拡張子パターンを結合
        all_images.extend(glob.glob(os.path.join(folder_path, ext)))
    return all_images

def display_random_images(image_paths, window_name, display_time_ms):
    """画像一覧からランダムに画像を選んで表示する"""

    if not image_paths:
        print("指定されたフォルダに画像ファイルが見つかりませんでした。")
        return

    # OpenCVのウィンドウを事前に作成し、サイズ変更を可能にする設定
    # WINDOW_NORMAL: ウィンドウのサイズを変更可能にする
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # WINDOW_GUI_EXPANDED: ディスプレイサイズに合わせて最大化する設定
    # NOTE: 環境によっては最大化されない場合もあります。
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"合計 {len(image_paths)} 枚の画像をランダムに表示します。（Escキーで終了）")

    while True:
        # 1. 画像をランダムに選択
        random_image_path = random.choice(image_paths)
        
        # 2. 画像を読み込む
        img = cv2.imread(random_image_path)
        
        if img is None:
            print(f"警告: 画像の読み込みに失敗しました: {random_image_path}")
            # 読み込めなかった画像をリストから除外して続行
            image_paths.remove(random_image_path)
            if not image_paths:
                print("表示可能な画像がなくなりました。")
                break
            continue
        
        # 3. 画像を表示
        cv2.imshow(window_name, img)
        print(f"表示中: {random_image_path}")
        
        # 4. 指定時間待機
        # cv2.waitKey() はキー入力を待つ関数。引数は待機時間(ms)。
        # Escキー (ASCIIコード 27)が押されたらループを抜ける
        key = cv2.waitKey(display_time_ms) & 0xFF
        if key == 27: 
            break
            
    # 全てのウィンドウを閉じる
    cv2.destroyAllWindows()

# --- メイン処理 ---
if __name__ == '__main__':
    # フォルダ内の画像パス一覧を取得
    image_list = get_image_list(FOLDER_PATH, EXTENSIONS)
    
    # 画像表示の実行
    display_random_images(image_list, WINDOW_NAME, DISPLAY_TIME_MS)