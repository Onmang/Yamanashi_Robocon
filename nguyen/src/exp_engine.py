#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO

model_path = "/home/konfi/robocon_yamanashi/Yamanashi_Robocon/nguyen/src/yolo_model/detect_5class_v8n/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format (FP16)
model.export(format="engine", half=True, device=0, workspace=2)
