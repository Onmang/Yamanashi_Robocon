# Yamanashi Robocon - Golf Robot

山梨大学ロボコンチームのゴルフ競技用ロボット開発用リポジトリです。

---

## 開発フロー
*   **メイン作業ブランチ**: 原則として `develop` ブランチで作業を行います。
*   **運用ルール**: 機能追加や修正は `develop` で行い、動作確認後にマージしてください。

---

## ディレクトリ構成
| ディレクトリ / ファイル | 説明 |
| :--- | :--- |
| `share_lib/` | チーム内で共有が必要な外部ライブラリ等を格納 |
| `.gitignore` | 実行ログ（`.log`）や大容量バイナリなど、Git管理不要なファイルを自動除外 |

---

## 開発環境の効率化設定（推奨）
ターミナル起動後の「ディレクトリ移動」と「環境設定（ROS 2 / Python仮想環境）」をコマンド一つで完了させるための設定手順です。

### 1. 移動・初期化スクリプトの準備
自作スクリプト用ディレクトリに、以下の設定ファイルを用意します。

**ファイルパス:** `~/scripts/move_robocon.sh`
```bash
#!/bin/bash
# 1. 目的のディレクトリへ移動
cd ~/Yamanashi_Robocon/nguyen/src

# 2. Python仮想環境の起動（必要に応じて解除）
# source ~/py38env/bin/activate

# 3. ROS 2 環境設定の読み込み
source /opt/ros/humble/setup.bash
# source ~/denso_ros2_ws/install/setup.bash

echo "Environment: Robocon Yamanashi (Python: \$(python3 --version))
```

### 2. エイリアス（ショートカット）の登録
どこからでも robocon と打つだけで上記スクリプトを実行できるよう、~/.bashrc にエイリアスを追記します。

```bash
# ~/.bashrc の末尾に追記
alias robocon='source ~/scripts/move_robocon.sh'
```

### 3. 設定の反映
以下のコマンドを実行して設定を有効化します。
```bash
source ~/.bashrc
```

これ以降、ターミナルで robocon と入力するだけで、適切なディレクトリへの移動と開発環境の構築が完了します。