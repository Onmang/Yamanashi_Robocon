#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load a YOLO26 model
<<<<<<< HEAD
model_path = "/home/konfi/Yamanashi_Robocon/nguyen/src/yolo_model/detect_5class_v8n/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26n.engine'
=======
model_path = "/home/konfi/robocon_yamanashi/Yamanashi_Robocon/nguyen/src/yolo_model/detect_5class_v8n/weights/best.pt"
model = YOLO(model_path)

# Export the model to TensorRT format
#model.export(format="onnx")  # creates 'yolo26n.onnx'


# Export the model to TensorRT format
model.export(format="engine", quantize=16, device=0, workspace=2)  # creates 'yolo26n.engine'
>>>>>>> e2afda8fee63aec3a17d7db47bcf8569e5674e4c
