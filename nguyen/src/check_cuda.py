#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA 環境の確認スクリプト
check_cuda.py
"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current device index: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("[-] CUDA is not available. YOLO will run on CPU.")