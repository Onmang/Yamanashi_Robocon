# TensorRT Python 3.8 バインディング ビルド手順(Jetson Nano / JetPack 4.6)

## 前提環境
- Jetson Nano, JetPack 4.6 (CUDA 10.2)
- Ubuntu 18.04 (bionic), aarch64
- TensorRT 8.2.3.0 (JetPack 4.6 付属)
- Python 3.8 (venv: `py38env`)
- `libpython3.8-dev` インストール済み(`/usr/include/python3.8/Python.h` が存在すること)

**注意:** `pip install tensorrt` は使わない。PyPIの`tensorrt`はx86_64/新CUDA向けで、Jetson(aarch64, CUDA10.2)には対応していない。

---

## 手順

### 1. 既存の壊れたtensorrtを削除
古いPython 3.6向けの`.pth`ファイルや`tensorrt`パッケージが残っているとpipエラーやImportErrorの原因になる。

```bash
sudo rm -rf /home/konfi/robocon_yamanashi/py38env/lib/python3.8/site-packages/tensorrt*
```

### 2. TensorRT OSSのブランチを正しいバージョンに合わせる
JetPack 4.6 = TensorRT 8.2系。`master`ブランチは新しすぎて非互換なので**必ず`release/8.2`を使う**。

```bash
cd ~/TensorRT
git checkout release/8.2
git submodule update --init --recursive
```

### 3. 古いビルドキャッシュを削除
ブランチ切り替え前のビルド残骸(`build/`)が残っていると不整合の原因になるため削除。

```bash
rm -rf ~/TensorRT/build
```

### 4. Python.hの確認(コピー不要、システムのものをそのまま使う)
`~/TensorRT/python3.8/include/`に既にPython.h一式が用意されていればOK。
中身が怪しい場合はシステムの正規ヘッダーで存在確認:

```bash
dpkg -L libpython3.8-dev | grep Python.h
# => /usr/include/python3.8/Python.h
```

### 5. pybind11をEXT_PATH配下にリンク
TensorRTのCMakeは`EXT_PATH`配下の`pybind11/include/pybind11/pybind11.h`を探すため、シンボリックリンクを作成。

```bash
ln -s ~/pybind11 ~/TensorRT/pybind11
```

### 6. ビルド実行
`python/build.sh`はNVIDIA公式のビルドスクリプト。環境変数で対象を指定して実行。

```bash
cd ~/TensorRT
TARGET_ARCHITECTURE=aarch64 \
CUDA_ROOT=/usr/local/cuda \
TRT_OSSPATH=~/TensorRT \
EXT_PATH=~/TensorRT \
PYTHON_MAJOR_VERSION=3 \
PYTHON_MINOR_VERSION=8 \
bash python/build.sh
```

成功すると以下にwheelが生成される:
```
~/TensorRT/python/build/dist/tensorrt-8.2.3.0-cp38-none-linux_aarch64.whl
```

### 7. インストール
**事前に手順1の削除を必ず実施しておくこと**(残っているとPermission deniedで失敗する)。

```bash
pip install ~/TensorRT/python/build/dist/tensorrt-8.2.3.0-cp38-none-linux_aarch64.whl
```

### 8. 動作確認

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
# => 8.2.3.0

python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print('TensorRT builder created OK')
"
```

---

## ハマりやすいポイント

| 問題 | 原因 | 対処 |
|---|---|---|
| `pip install tensorrt`が失敗 | PyPI版はJetson非対応 | OSSから自前ビルド |
| `Python version mismatch: compiled for Python 3.6` | 古いtensorrt(3.6版)が残存 | 完全削除してから再インストール |
| `tensorrt_version.pth`でImportError | 古い`.pth`が起動時に毎回評価される | 該当`.pth`ファイルを削除 |
| `Permission denied`でpip installが失敗 | 既存ファイルの所有者/権限問題 | `sudo rm -rf`で完全削除後に再実行 |
| ビルド後`__version__`属性がない | ビルド設定/バージョン不一致 | `release/8.2`ブランチ・正しいCMake変数で再ビルド |

---

## 次のステップ
- Ultralytics(YOLOv8)のTensorRTエクスポート/推論確認
- PyTorch(`torch-1.10.0-cp38`)がPython 3.6バイナリ問題を抱えている件の確認・再ビルド
