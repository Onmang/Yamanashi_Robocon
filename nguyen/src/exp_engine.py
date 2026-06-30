#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load a YOLO26 model
model_path = "/home/konfi/Yamanashi_Robocon/nguyen/src/yolo_model/detect_5class_v8n/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26n.engine'