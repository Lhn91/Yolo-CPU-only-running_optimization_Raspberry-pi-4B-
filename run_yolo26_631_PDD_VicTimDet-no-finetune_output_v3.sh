#!/bin/bash
# Script để khởi chạy nhanh model yolo26_631_PDD_VicTimDet-no-finetune_output_v3
# Nhớ cấp quyền thực thi trên Pi: chmod +x run_yolo26_631_PDD_VicTimDet-no-finetune_output_v3.sh

# Chuyển hướng làm việc vào thư mục EdgeVisionRT hiện tại
cd "$(dirname "$0")"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm

echo "Cap quyen Framebuffer cho LCD..."
sudo chmod 666 /dev/fb0

echo "Kich hoat model yolo26_631_PDD_VicTimDet-no-finetune_output_v3 qua Webcam + Framebuffer + INT8..."
# Truyền đúng tên base của model (bao gồm cả .ncnn để đúng với file .ncnn.param)
./run.sh cam fb int8 model "yolo26_631_PDD_VicTimDet-no-finetune_output_v3"
