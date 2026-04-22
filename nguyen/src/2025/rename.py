import os
import re

# 画像ファイルがあるディレクトリ
target_dir = "."

for filename in os.listdir(target_dir):
    old_path = os.path.join(target_dir, filename)

    # ファイルでなければスキップ
    if not os.path.isfile(old_path):
        continue

    # 正規表現で (数字) を _数字 に変換
    new_name = re.sub(r"\((\d+)\)", r"_\1", filename)

    # "Image" を小文字にして "image" に揃える
    new_name = new_name.replace("Image", "image")

    new_path = os.path.join(target_dir, new_name)

    # 名前が変わる場合のみリネーム
    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
