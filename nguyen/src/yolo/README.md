# YOLO Object Detection Project

このプロジェクトは、YOLOv8/YOLOv11を使用した物体検出（ボール検出など）のための学習・推論環境です。

## ディレクトリ構成

### 📂 datasets/
データセットを格納するフォルダ
- `images_val/`: 検証用画像データ
- その他の学習・検証データ

### 📂 runs/
学習結果と検出結果を保存するフォルダ
- **学習結果（例: `ball_detect_v22/`）**
  - `results.csv`: 学習の精度推移データ
  - `results.png`: 学習グラフ（損失、精度など）
  - `confusion_matrix.png`: 混同行列
  - `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`: 評価指標のグラフ
  - `args.yaml`: 学習時のハイパーパラメータ
  - `train_batch*.jpg`: 学習バッチのサンプル画像
  - `val_batch*_labels.jpg`: 検証バッチの予測結果
  - `weights/`: 学習済みモデル（best.pt, last.ptなど）

- **検出結果（例: `detect/`）**
  - 推論実行時の検出結果画像が保存される

### 📂 auto_ann/
自動アノテーション結果を格納するフォルダ
- `predict_labels/`: 学習済みモデルを使った自動ラベリングの結果
- `README.txt`: フォルダの説明

### 📄 モデルファイル
- `yolo11n.pt`: YOLOv11 nano版の事前学習モデル
- `yolov8n.pt`: YOLOv8 nano版の事前学習モデル

### 📓 ノートブック
- `yolo_test.ipynb`: YOLOテスト用ノートブック 1
- `yolo_test_2.ipynb`: YOLOテスト用ノートブック 2
- `yolo_test_3.ipynb`: YOLOテスト用ノートブック 3

## 学習方法

本プロジェクトでは、2通りの学習方法を使用しています。

### 1. Google Colabでの学習
- **環境**: Google Colaboratory（GPU使用）
- **データセット**: Google Drive上に保存
- **特徴**: クラウド環境で学習を実行、GPUが無料で利用可能
- **手順**:
  1. Google Driveにデータセットをアップロード
  2. Colabノートブックでドライブをマウント
  3. YOLOモデルをトレーニング
  4. 学習結果をDriveに保存

### 2. ローカルPCでの学習（注湯部屋PC）
- **環境**: 注湯部屋のローカルPC
- **データセット**: `C:\Users\guenk\yolo\datasets`
- **特徴**: ローカル環境で学習、ネットワーク接続不要
- **手順**:
  1. データセットを上記パスに配置
  2. ローカル環境でYOLOモデルをトレーニング
  3. 学習結果は `runs/` フォルダに自動保存

## ラベリング方法

### Roboflow
基本的なラベリング作業には **Roboflow** を使用しています。

- **URL**: https://roboflow.com/
- **機能**:
  - Webベースのアノテーションツール
  - データ拡張（Augmentation）機能
  - データセットのバージョン管理
  - エクスポート形式：YOLO形式、COCO形式など

**ワークフロー**:
1. Roboflowにプロジェクトを作成
2. 画像をアップロード
3. バウンディングボックスでアノテーション
4. データ拡張の設定（オプション）
5. YOLO形式でエクスポート
6. ダウンロードしたデータセットを学習環境に配置

## 使用方法

### 学習の実行
```python
from ultralytics import YOLO

# モデルのロード
model = YOLO('yolo11n.pt')

# 学習の実行
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 推論の実行
```python
from ultralytics import YOLO

# 学習済みモデルのロード
model = YOLO('runs/ball_detect_v22/weights/best.pt')

# 推論
results = model.predict(source='path/to/image.jpg', save=True)
```

## 参考リンク
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)

## 備考
- 学習結果は自動的に `runs/` フォルダ内にタイムスタンプ付きで保存されます
- 最適なモデルは `best.pt`、最後のエポックのモデルは `last.pt` として保存されます
- GPUを使用する場合、CUDAとcuDNNが正しくインストールされていることを確認してください
