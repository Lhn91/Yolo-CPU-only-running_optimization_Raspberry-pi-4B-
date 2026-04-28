#!/bin/bash
# Script để khởi chạy nhanh model Yolo26s_stretch_47k_631_PDD_VicTimDet_No_TV_weather_output
# Nhớ cấp quyền thực thi trên Pi: chmod +x run_Yolo26s_stretch_47k_631_PDD_VicTimDet_No_TV_weather_output.sh

# Chuyển hướng làm việc vào thư mục EdgeVisionRT hiện tại
cd "$(dirname "$0")"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm

echo "Cap quyen Framebuffer cho LCD..."
sudo chmod 666 /dev/fb0

echo "Kich hoat model Yolo26s_stretch_47k_631_PDD_VicTimDet_No_TV_weather_output qua Webcam + Framebuffer + INT8..."
# Truyền đúng tên base của model (bao gồm cả .ncnn để đúng với file .ncnn.param)
./run.sh cam fb int8 model "Yolo26s_stretch_47k_631_PDD_VicTimDet_No_TV_weather_output"
