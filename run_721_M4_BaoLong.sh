#!/bin/bash
# Script để khởi chạy nhanh model 721_M4_BaoLong.ncnn
# Nhớ cấp quyền thực thi trên Pi: chmod +x run_721_M4_BaoLong.sh

# Chuyển hướng làm việc vào thư mục EdgeVisionRT hiện tại
cd "$(dirname "$0")"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm

echo "Cap quyen Framebuffer + Input devices cho LCD..."
sudo chmod 666 /dev/fb0
sudo chmod 666 /dev/input/event*

# Setup PWM cho Servo MG996R (GPIO 12)
# Yêu cầu: thêm "dtoverlay=pwm,pin=12,func=4" vào /boot/config.txt và reboot
if [ -d /sys/class/pwm/pwmchip0 ]; then
    echo 0 | sudo tee /sys/class/pwm/pwmchip0/export > /dev/null 2>&1 || true
    sleep 0.1
    sudo chmod 666 /sys/class/pwm/pwmchip0/pwm0/* 2>/dev/null || true
    echo "PWM servo: ready"
fi

echo "Kich hoat model 721_M4_BaoLong.ncnn: Webcam + Framebuffer + INT8 + Servo + Dashboard (run.sh)..."
# Khớp run.sh mới: thêm "dashboard" để gửi dữ liệu lên Web Dashboard (mặc định http://127.0.0.1:8000 trong run.sh)
# Tên model: bao gồm .ncnn để khớp 721_M4_BaoLong.ncnn.param / .bin
# Video full (Tailscale): run.sh bật MJPEG tích hợp :8080 (như Coral). Cài Tailscale trên Pi + laptop;
# binary tự POST /stream-config với http://<100.x|LAN>:8080/stream. Tắt MJPEG: MJPEG_PORT=0
./run.sh cam fb int8 servo dashboard model 721_M4_BaoLong.ncnn
