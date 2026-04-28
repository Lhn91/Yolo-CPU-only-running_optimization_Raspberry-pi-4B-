#!/bin/bash
# Script để khởi chạy nhanh model yolov8n_int8
# Nhớ cấp quyền thực thi trên Pi: chmod +x run_yolov8n_int8.sh

# Chuyển hướng làm việc vào thư mục EdgeVisionRT hiện tại
cd "$(dirname "$0")"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm

echo "Cap quyen Framebuffer + Input devices cho LCD..."
sudo chmod 666 /dev/fb0
sudo chmod 666 /dev/input/event*

echo "Kich hoat model yolov8n_int8 qua Webcam + Framebuffer..."
# Truyền đúng tên thư mục chứa model
./run.sh cam fb int8 model "yolov8n_int8_ncnn_model"
