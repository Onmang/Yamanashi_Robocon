#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_ball_test.py

yolo学習済みモデルでリアルタイム物体検出

Example:
    $ ./detect_ball_test.py  # after run: chmod +x detect_ball_test.py
    $ python realsense_yolov8.py --model C:\\Users\\guenk\\Yamanashi_Robocon\\nguyen\\src\\yolo\\runs\\ball_detect_v22\\weights\\best.pt

Author:
    nguyen

Date:
    2026-03-05

Version:
    1.0.0

Requirements:
    - pip install ultralytics pyrealsense2 opencv-python
    - python >= 3.9     https://www.python.org/
    - ultralytics >= 8.4.55     https://github.com/ultralytics/ultralytics
    - pyrealsense2 >= 2.56.5.9235   https://pypi.org/project/pyrealsense2/
    - opencv-python >= 4.13.0.92    https://pypi.org/project/opencv-python/

History:
    - 2026-03-05: nguyen -新規作成
    - 2026-06-06: nguyen -ヘッダー変更
"""

import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# [parse_args] 簡潔な役割説明
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def parse_args():
    """
    コマンドライン引数を解析する
    Returns:
        argparse.Namespace: 解析された引数の名前空間    
        
    """
    parser = argparse.ArgumentParser(description="YOLOv8 + RealSense リアルタイム検出")
    parser.add_argument("--model", type=str, default="C:\\Users\\guenk\\Yamanashi_Robocon\\nguyen\\src\\yolo\\runs\\ball_detect_v22\\weights\\best.pt", help="YOLOv8モデルパス")
    parser.add_argument("--conf", type=float, default=0.5, help="信頼度閾値 (0~1)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU閾値")
    parser.add_argument("--width", type=int, default=640, help="カメラ解像度 幅")
    parser.add_argument("--height", type=int, default=480, help="カメラ解像度 高さ")
    parser.add_argument("--fps", type=int, default=30, help="フレームレート")
    parser.add_argument("--show-depth", action="store_true", help="深度マップも表示する")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# [setup_realsense] RealSense パイプラインを初期化する
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def setup_realsense(width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align]:
    """
    RealSense D405 カメラのパイプラインを初期化する
    Args:
        width (int): カメラ解像度の幅
        height (int): カメラ解像度の高さ
        fps (int): フレームレート
    Returns:
        tuple[rs.pipeline, rs.align]: 初期化されたパイプラインとアラインオブジェクト
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # カラーストリーム
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # 深度ストリーム
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    pipeline.start(config)

    # 深度フレームをカラーフレームに位置合わせ
    align = rs.align(rs.stream.color)

    # 自動露光が安定するまで数フレーム捨てる
    for _ in range(5):
        pipeline.wait_for_frames()

    print("[INFO] RealSense D405 初期化完了")
    return pipeline, align

# -----------------------------------------------------------------------------
# [get_depth_at_bbox] バウンディングボックス中心の深度 [mm] を取得する
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def get_depth_at_bbox(depth_frame, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    バウンディングボックス中心の深度 [mm] を取得する
    Args:
        depth_frame: 深度フレーム
        x1 (int): バウンディングボックスの左上X座標
        y1 (int): バウンディングボックスの左上Y座標
        x2 (int): バウンディングボックスの右下X座標
        y2 (int): バウンディングボックスの右下Y座標
    Returns:
        float: 深度 [mm]
    """
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth_value = depth_frame.get_distance(cx, cy)  # メートル単位
    return depth_value * 1000  # mm に変換

# -----------------------------------------------------------------------------
# [draw_detections] 検出結果と深度情報を画像に描画す
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def draw_detections(color_image: np.ndarray, results, depth_frame) -> np.ndarray:
    """
    検出結果と深度情報を画像に描画する
    Args:
        color_image (np.ndarray): カラー画像
        results: YOLOv8の推論結果
        depth_frame: 深度フレーム
    Returns:
        np.ndarray: 描画された画像
    """
    annotated = color_image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # 座標・スコア・クラス取得
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            # 深度取得
            depth_mm = get_depth_at_bbox(depth_frame, x1, y1, x2, y2)
            depth_str = f"{depth_mm:.0f}mm" if depth_mm > 0 else "N/A"

            # バウンディングボックス描画
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ラベル
            label = f"{cls_name} {conf:.2f} | {depth_str}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx, ly = x1, max(y1 - 10, label_size[1])
            cv2.rectangle(
                annotated,
                (lx, ly - label_size[1] - 4),
                (lx + label_size[0], ly + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

    return annotated

# -----------------------------------------------------------------------------
# [main] 
# -----------------------------------------------------------------------------
# Updates:
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # モデル読み込み
    print(f"[INFO] モデル読み込み中: {args.model}")
    model = YOLO(args.model)

    # RealSense 初期化
    pipeline, align = setup_realsense(args.width, args.height, args.fps)

    # 深度カラーマップ用
    colorizer = rs.colorizer()

    print("[INFO] 検出開始。終了するには 'q' キーを押してください。")

    try:
        while True:
            # フレーム取得
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # YOLOv8 推論
            results = model.predict(
                source=color_image,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )

            # 描画
            annotated_image = draw_detections(color_image, results, depth_frame)

            # FPS 表示
            fps_text = f"FPS: {1 / (results[0].speed['inference'] / 1000 + 1e-9):.1f}"
            cv2.putText(
                annotated_image, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )

            # カラー画像表示
            cv2.imshow("YOLOv8 + RealSense D405", annotated_image)

            # 深度マップ表示（オプション）
            if args.show_depth:
                depth_colormap = np.asanyarray(
                    colorizer.colorize(depth_frame).get_data()
                )
                cv2.imshow("Depth Map", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 終了しました。")


if __name__ == "__main__":
    main()