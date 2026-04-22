# -*- coding: utf-8 -*-
# 共通関数, 主に描画用

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs 

from common_function import (
    compute_center_distance,
)

# -*- coding: utf-8 -*-
# common_function_vis.py
# 描画まわりの共通処理

import cv2
import numpy as np

from common_function import compute_center_distance


def draw_center_distance_debug(color_image, depth_image, cam, W=640, H=480, roi_size=20):
    """
    画像中心の距離をオーバーレイ表示するだけの簡単なやつ
    """
    cx, cy = W // 2, H // 2
    center_dist_m, center_dist_mm, _, (x1, y1), (x2, y2) = compute_center_distance(
        depth_image,
        cam.depth_scale,
        W,
        H,
        cx,
        cy,
        roi_size=roi_size,
    )

    overlay = color_image.copy()
    dist_text = f"Center: {center_dist_m:.3f} [m] ({center_dist_mm:.0f} [mm])"

    cv2.putText(
        overlay,
        dist_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 1)

    return overlay


def draw_circularity_candidates(
    mask_morph,
    vis,
    labels,
    stats,
    centroids,
    area_min=100,
    circ_min=0.80,
):
    """
    「円形度を見て候補だけ色つける」パートを関数化
    戻り値:
        candidate_mask : 円形度OKなラベルだけ255にしたマスク (gray)
        mask_dbg       : ラベルを四角と文字で描いたデバッグ画像 (BGR)
        vis_dbg        : 元のvisに円を描いたやつ (BGR)
    """
    # デバッグ用にカラーにしておく
    mask_dbg = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
    vis_dbg = vis.copy()

    h, w = mask_morph.shape[:2]
    candidate_mask = np.zeros((h, w), dtype=np.uint8)

    # i=0 は背景なので 1 から
    for i in range(1, len(stats)):
        x, y, ww, hh, area = stats[i]
        cx, cy = int(centroids[i][0]), int(centroids[i][1])

        if area < area_min:
            continue

        # ラベルだけ抜いたマスク
        blob_mask = np.zeros_like(mask_morph)
        blob_mask[labels == i] = 255

        # その領域だけで輪郭をとる
        roi = blob_mask[y : y + hh, x : x + ww]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_round_enough = False
        for cnt in contours:
            area_cnt = cv2.contourArea(cnt)
            if area_cnt <= 0:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = (4.0 * np.pi * area_cnt) / (peri * peri)

            if circularity >= circ_min:
                is_round_enough = True

                # デバッグ描画（矩形＋中心＋文字）
                cv2.rectangle(mask_dbg, (x, y), (x + ww, y + hh), (255, 0, 0), 2)
                cv2.circle(mask_dbg, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(
                    mask_dbg,
                    f"[{i}]:{area} C:{circularity:.2f}",
                    (x + 5, y - 5 if y - 5 > 10 else y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # 円としても描いとく（元のvis側）
                (cx_f, cy_f), r_f = cv2.minEnclosingCircle(cnt)
                cx_abs = int(cx_f) + x
                cy_abs = int(cy_f) + y
                r_px = int(r_f)
                cv2.circle(vis_dbg, (cx_abs, cy_abs), r_px, (255, 0, 255), 2)
                cv2.circle(vis_dbg, (cx_abs, cy_abs), 2, (255, 255, 0), 3)
                cv2.putText(
                    vis_dbg,
                    f"C:{circularity:.2f}",
                    (cx_abs + 30, cy_abs + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                break  # このラベルはOKなので次のラベルへ

        if is_round_enough:
            candidate_mask[labels == i] = 255

    return candidate_mask, mask_dbg, vis_dbg


def draw_flag_triangles_debug(
    bgr,
    mask_morph,
    tri_area_min=500,
    approx_eps_ratio=0.1,
):
    """
    HSVマスクから「三角形=旗」を描画するパートを関数化
    戻り値:
        vis        : 描画済みBGR
        triangles  : 見つかった三角形のリスト
    """
    # マスクをかけた表示用
    vis = cv2.bitwise_and(bgr, bgr, mask=mask_morph)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    approx_contours = []
    for cnt in contours:
        arclen = cv2.arcLength(cnt, True)
        approx_cnt = cv2.approxPolyDP(cnt, epsilon=approx_eps_ratio * arclen, closed=True)
        approx_contours.append(approx_cnt)

    # 三角形だけ抜き出す
    triangles = list(filter(lambda x: len(x) == 3, approx_contours))
    cv2.drawContours(vis, triangles, -1, (0, 0, 255), 2)  # 赤で三角だけ

    # 面積と中心を描く
    for tri in triangles:
        area = cv2.contourArea(tri)
        if area < tri_area_min:
            continue

        x, y, w, h = cv2.boundingRect(tri)
        cv2.putText(
            vis,
            f"Tri:{area:.0f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        M = cv2.moments(tri)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

    return vis, triangles





