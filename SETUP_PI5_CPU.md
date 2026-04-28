# HƯỚNG DẪN CÀI ĐẶT YOLOV8 C++ EDGEVISIONRT BẰNG CHUẨN NCNN TRÊN RASPBERRY PI 5 (CPU MODE)

Tài liệu này tóm tắt cách tải mã nguồn C++, biên dịch và chạy cấu hình tối ưu sức mạnh đa luồng CPU (ARMv8 Cortex-A76) siêu hiệu năng của con Raspberry Pi 5 thông qua thư viện mạng NCNN của Tencent. Chế độ này thường không cần đến phần cứng hỗ trợ bên ngoài.

## BƯỚC 1: CÀI ĐẶT CÁC CÔNG CỤ COMPILER CỦA LINUX

Để biên dịch được bộ mã nguồn C/C++ EdgeVisionRT, Raspberry Pi 5 cần có CMake và bộ phát triển OpenCV tối thiểu.

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential g++ \
libopencv-dev libvulkan-dev vulkan-tools
```

---

## BƯỚC 2: BUILD DỰ ÁN EDGEVISION-RT NCNN

EdgeVisionRT đã chứa sẵn cấu trúc để tích hợp thẳng thư viện `ncnn`.

```bash
# 1. Di chuyển vào thư mục Source Code
cd ~/EdgeVisionRT

# 2. Dọn sạch bản build bị lỗi (Nếu có)
rm -rf build
mkdir build
cd build

# 3. Cấu hình CMake cho Môi trường Raspberry Pi 5 (Thường chỉ CPU, chưa cần Vulkan nếu không muốn lỗi Driver)
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF ..

# 4. Bắt đầu build với 4 luồng Core (Pi 5 có 4 Core vật lý, làm việc cực nhanh)
make -j4
```
*(Khi thanh tiến trình chạy 100% không báo màu Đỏ, Project C++ đã ra lò file nhị phân `EdgeVisionRT`)*

---

## BƯỚC 3: XUẤT MÔ HÌNH YOLO SANG ĐỊNH DẠNG NCNN

NCNN framework trong C++ yêu cầu 2 file model đặc thù là `.param` (Kiến trúc) và `.bin` (Trọng số). Bạn chuyển đổi trực tiếp con mô hình PyTorch `.pt` của bạn bằng máy Windows / Ubuntu có cài thư viện `ultralytics`.

```python
# Cài thư viện export nhanh ncnn
pip install ultralytics ncnn

# Dùng Python console hoặc Bash:
yolo export model=yolov8n.pt format=ncnn imgsz=416
```
Sau đó hệ thống sẽ sinh ra một thư mục `yolov8n_ncnn_model`. 
Copy thư mục này sang Raspberry Pi 5 của bạn và đặt vào `~/EdgeVisionRT/models/`.

---

## BƯỚC 4: CHỈNH SỬA CẤU HÌNH VÀ CHẠY THỰC TẾ

EdgeVisionRT được tối ưu cho Raspberry cực tốt, chạy đa luồng Camera riêng, hiển thị riêng. 

1. **Sửa file config nếu chỉ huấn luyện 1 Class (`nguoi`):**
Mở file `include/common.h` và kiếm dòng số lượng Class để cập nhật.
```cpp
// Trong file EdgeVisionRT/include/common.h
#define NUM_CLASSES 1 
```
*(Lưu ý: Sau khi sửa Header C++, bạn phải quay lại thư mục build gõ lệnh `make -j4` lại để nó nạp lại Rule).*

2. **Chạy inference với Camera USB /dev/video0:**
Chúng ta có thể vẽ thẳng lên Framebuffer (`/dev/fb0`) tương tự giải pháp Python.

```bash
cd ~/EdgeVisionRT

# Chạy bản build C++
sudo ./build/EdgeVisionRT --model models/yolov8n_ncnn_model --source /dev/video0
```

Raspberry Pi 5 với xung nhịp 2.4GHz hoàn toàn cày mượt kiến trúc NCNN INT8/FP16 trên CPU đạt mức dao động 10 - 20 FPS cho tệp ảnh kích cỡ `416x416` tuỳ biến!
