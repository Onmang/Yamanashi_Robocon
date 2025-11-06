import cv2
import numpy as np
from matplotlib import pyplot as plt

# 2値画像を読み込み
bin_img = cv2.imread("hsv_img.png", cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(bin_img, 1, 255, cv2.THRESH_BINARY)

h, w = bin_img.shape

# 描画用にBGRに変換
vis = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

# ===== 方法①: 下部の帯領域から重心を取る =====
band_h = 40
roi = bin_img[h-band_h : h, :]
contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    c1 = max(contours, key=cv2.contourArea)
    xs = c1[:,0,0]
    x_center_local = int(xs.mean())
    x_center1 = x_center_local
    y_center1 = h - band_h//2
    cv2.circle(vis, (x_center1, y_center1), 5, (0,0,255), -1)
    cv2.putText(vis, "M1", (x_center1+5, y_center1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# ===== 方法②: 輪郭の下端付近の点から重心を取る =====
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    c2 = max(contours, key=cv2.contourArea)
    pts = c2[:,0,:]
    ys = pts[:,1]
    y_max = ys.max()
    thresh = 15
    mask = ys > (y_max - thresh)
    bottom_pts = pts[mask]
    x_center2 = int(bottom_pts[:,0].mean())
    y_center2 = int(bottom_pts[:,1].mean())
    cv2.circle(vis, (x_center2, y_center2), 5, (255,0,0), -1)
    cv2.putText(vis, "M2", (x_center2+5, y_center2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

plt.figure(figsize=(4,3))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
