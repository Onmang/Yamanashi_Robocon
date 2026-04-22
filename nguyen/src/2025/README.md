## 主要ファイル

### 1. メイン認識プログラム

- `recognition_ver1.py`: 一般テスト、パラメータ設定用
- `recognition_ver2.py`: メインテスト、ボールを追っかける
- `find_green_centroid.py`: 緑色の物体の重心を検出。
- `recognition_flag.py`: 旗の認識用

### 2. テスト・調整用スクリプト

- `hsv_test.py`: 色検出（HSV）の閾値調整用 GUI ツール。
- `hough_circles_only.py`: ハフ変換による円検出のテスト用。
- `d405_view.py`, `rgbd_camera_test.py`: RealSense カメラの動作確認用。
- `change_camera_test.py`: カメラ切り替えテスト用

### 3. 設定ファイル (`.json`)

- `hsv_params_*.json`: 色検出用の HSV 閾値設定。
- `houghcircles_*.json`: 円検出のパラメータ設定。
- `distance_*.json`: 距離フィルタリングのパラメータ設定。
- `filter_params.json`, `gaussian_filter_params.json`: 各種画像処理フィルタのパラメータ。

### 4. 共通モジュール・その他

- `common_function.py`: 複数のスクリプトから呼び出される共通関数群。
