#!/bin/bash
# Script để khởi chạy nhanh model Yolov11_n
# Nhớ cấp quyền thực thi trên Pi: chmod +x run_Yolov11_n.sh

# Chuyển hướng làm việc vào thư mục EdgeVisionRT hiện tại
cd "$(dirname "$0")"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm

echo "Cap quyen Framebuffer + Input devices cho LCD..."
sudo chmod 666 /dev/fb0
sudo chmod 666 /dev/input/event*

echo "Kich hoat model Yolov11_n qua Webcam + Framebuffer + INT8..."
# Truyền đúng tên base của model (bao gồm cả .ncnn để đúng với file .ncnn.param)
./run.sh cam fb int8 model "Yolov11_n"
