# -*- coding: utf-8 -*-
"""path_config.py

path maneger

Author:
    nguyen

Date:
    2026-07-05

History:
    - 2026-07-05: nguyen coped from 2025
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# path for parameters
PARAM_PATH_DIS_LIMIT = str(BASE_DIR / "distance_params.json")

# YOLO model, jetson nano
YOLO_MODEL = str(BASE_DIR / "yolo_model"/ "detect_5class_v8n"/ "weights"/ "best.engine")