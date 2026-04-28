/**
 * @file drm_display.h
 * @brief Direct Framebuffer display — no X11, full-screen scale, class labels
 *
 * FEATURES:
 *  - Scale any input frame to full screen (nearest-neighbor)
 *  - 16bpp (RGB565) and 32bpp (BGRA) support
 *  - Per-class color bounding boxes (20 distinct colors, cycles over 80 COCO classes)
 *  - Class name + confidence % label with black background for readability
 *  - Embedded 8x8 bitmap font (full printable ASCII 32-126)
 *
 * USAGE: sudo systemctl stop lightdm && sudo chmod 666 /dev/fb0
 */

#ifndef YOLO_DRM_DISPLAY_H
#define YOLO_DRM_DISPLAY_H

#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <functional>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <arm_neon.h>

namespace yolo {

// ============================================================================
// 8x8 Bitmap Font  (printable ASCII 32–126, 95 chars × 8 bytes)
// Standard VGA/PC 8x8 font — each byte = one row, MSB = leftmost pixel
// ============================================================================
static const uint8_t FONT8x8[95][8] = {
/*32 ' '*/  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
/*33 '!'*/  {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00},
/*34 '"'*/  {0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00},
/*35 '#'*/  {0x6C,0x6C,0xFE,0x6C,0xFE,0x6C,0x6C,0x00},
/*36 '$'*/  {0x18,0x3E,0x60,0x3C,0x06,0x7C,0x18,0x00},
/*37 '%'*/  {0x00,0xC6,0xCC,0x18,0x30,0x66,0xC6,0x00},
/*38 '&'*/  {0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00},
/*39 '\''*/ {0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00},
/*40 '('*/  {0x0E,0x1C,0x18,0x18,0x18,0x1C,0x0E,0x00},
/*41 ')'*/  {0x70,0x38,0x18,0x18,0x18,0x38,0x70,0x00},
/*42 '*'*/  {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00},
/*43 '+'*/  {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00},
/*44 ','*/  {0x00,0x00,0x00,0x00,0x18,0x18,0x30,0x00},
/*45 '-'*/  {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00},
/*46 '.'*/  {0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00},
/*47 '/'*/  {0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00},
/*48 '0'*/  {0x7C,0xC6,0xCE,0xDE,0xF6,0xE6,0x7C,0x00},
/*49 '1'*/  {0x18,0x38,0x68,0x18,0x18,0x18,0x7E,0x00},
/*50 '2'*/  {0x7C,0xC6,0x06,0x1C,0x30,0x66,0xFE,0x00},
/*51 '3'*/  {0x7C,0xC6,0x06,0x3C,0x06,0xC6,0x7C,0x00},
/*52 '4'*/  {0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x1E,0x00},
/*53 '5'*/  {0xFE,0xC0,0xC0,0xFC,0x06,0xC6,0x7C,0x00},
/*54 '6'*/  {0x38,0x60,0xC0,0xFC,0xC6,0xC6,0x7C,0x00},
/*55 '7'*/  {0xFE,0xC6,0x0C,0x18,0x30,0x30,0x30,0x00},
/*56 '8'*/  {0x7C,0xC6,0xC6,0x7C,0xC6,0xC6,0x7C,0x00},
/*57 '9'*/  {0x7C,0xC6,0xC6,0x7E,0x06,0x0C,0x78,0x00},
/*58 ':'*/  {0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00},
/*59 ';'*/  {0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x00},
/*60 '<'*/  {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00},
/*61 '='*/  {0x00,0x66,0x66,0x00,0x66,0x66,0x00,0x00},
/*62 '>'*/  {0x60,0x30,0x18,0x0C,0x18,0x30,0x60,0x00},
/*63 '?'*/  {0x7C,0xC6,0x0C,0x18,0x18,0x00,0x18,0x00},
/*64 '@'*/  {0x7C,0xC6,0xDE,0xDE,0xDE,0xC0,0x78,0x00},
/*65 'A'*/  {0x30,0x78,0xCC,0xCC,0xFC,0xCC,0xCC,0x00},
/*66 'B'*/  {0xFC,0x66,0x66,0x7C,0x66,0x66,0xFC,0x00},
/*67 'C'*/  {0x3C,0x66,0xC0,0xC0,0xC0,0x66,0x3C,0x00},
/*68 'D'*/  {0xF8,0x6C,0x66,0x66,0x66,0x6C,0xF8,0x00},
/*69 'E'*/  {0xFE,0x62,0x68,0x78,0x68,0x62,0xFE,0x00},
/*70 'F'*/  {0xFE,0x62,0x68,0x78,0x68,0x60,0xF0,0x00},
/*71 'G'*/  {0x3C,0x66,0xC0,0xC0,0xCE,0x66,0x3E,0x00},
/*72 'H'*/  {0xCC,0xCC,0xCC,0xFC,0xCC,0xCC,0xCC,0x00},
/*73 'I'*/  {0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},
/*74 'J'*/  {0x1E,0x0C,0x0C,0x0C,0xCC,0xCC,0x78,0x00},
/*75 'K'*/  {0xE6,0x66,0x6C,0x78,0x6C,0x66,0xE6,0x00},
/*76 'L'*/  {0xF0,0x60,0x60,0x60,0x62,0x66,0xFE,0x00},
/*77 'M'*/  {0xC6,0xEE,0xFE,0xFE,0xD6,0xC6,0xC6,0x00},
/*78 'N'*/  {0xC6,0xE6,0xF6,0xDE,0xCE,0xC6,0xC6,0x00},
/*79 'O'*/  {0x38,0x6C,0xC6,0xC6,0xC6,0x6C,0x38,0x00},
/*80 'P'*/  {0xFC,0x66,0x66,0x7C,0x60,0x60,0xF0,0x00},
/*81 'Q'*/  {0x78,0xCC,0xCC,0xCC,0xDC,0x78,0x1C,0x00},
/*82 'R'*/  {0xFC,0x66,0x66,0x7C,0x6C,0x66,0xE6,0x00},
/*83 'S'*/  {0x78,0xCC,0xE0,0x70,0x1C,0xCC,0x78,0x00},
/*84 'T'*/  {0x7E,0x7E,0x5A,0x18,0x18,0x18,0x3C,0x00},
/*85 'U'*/  {0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0xFC,0x00},
/*86 'V'*/  {0xCC,0xCC,0xCC,0xCC,0xCC,0x78,0x30,0x00},
/*87 'W'*/  {0xC6,0xC6,0xC6,0xD6,0xFE,0xEE,0xC6,0x00},
/*88 'X'*/  {0xC6,0xC6,0x6C,0x38,0x38,0x6C,0xC6,0x00},
/*89 'Y'*/  {0x66,0x66,0x66,0x3C,0x18,0x18,0x3C,0x00},
/*90 'Z'*/  {0xFE,0xC6,0x8C,0x18,0x32,0x66,0xFE,0x00},
/*91 '['*/  {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00},
/*92 '\\'*/ {0xC0,0x60,0x30,0x18,0x0C,0x06,0x02,0x00},
/*93 ']'*/  {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00},
/*94 '^'*/  {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00},
/*95 '_'*/  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF},
/*96 '`'*/  {0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00},
/*97 'a'*/  {0x00,0x00,0x78,0x0C,0x7C,0xCC,0x76,0x00},
/*98 'b'*/  {0xE0,0x60,0x60,0x7C,0x66,0x66,0xDC,0x00},
/*99 'c'*/  {0x00,0x00,0x78,0xCC,0xC0,0xCC,0x78,0x00},
/*100 'd'*/ {0x1C,0x0C,0x0C,0x7C,0xCC,0xCC,0x76,0x00},
/*101 'e'*/ {0x00,0x00,0x78,0xCC,0xFC,0xC0,0x78,0x00},
/*102 'f'*/ {0x38,0x6C,0x60,0xF0,0x60,0x60,0xF0,0x00},
/*103 'g'*/ {0x00,0x00,0x76,0xCC,0xCC,0x7C,0x0C,0xF8},
/*104 'h'*/ {0xE0,0x60,0x6C,0x76,0x66,0x66,0xE6,0x00},
/*105 'i'*/ {0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00},
/*106 'j'*/ {0x06,0x00,0x06,0x06,0x06,0x66,0x66,0x3C},
/*107 'k'*/ {0xE0,0x60,0x66,0x6C,0x78,0x6C,0xE6,0x00},
/*108 'l'*/ {0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},
/*109 'm'*/ {0x00,0x00,0xCC,0xFE,0xFE,0xD6,0xC6,0x00},
/*110 'n'*/ {0x00,0x00,0xF8,0xCC,0xCC,0xCC,0xCC,0x00},
/*111 'o'*/ {0x00,0x00,0x78,0xCC,0xCC,0xCC,0x78,0x00},
/*112 'p'*/ {0x00,0x00,0xDC,0x66,0x66,0x7C,0x60,0xF0},
/*113 'q'*/ {0x00,0x00,0x76,0xCC,0xCC,0x7C,0x0C,0x1E},
/*114 'r'*/ {0x00,0x00,0xDC,0x76,0x66,0x60,0xF0,0x00},
/*115 's'*/ {0x00,0x00,0x7C,0xC0,0x78,0x0C,0xF8,0x00},
/*116 't'*/ {0x10,0x30,0x7C,0x30,0x30,0x34,0x18,0x00},
/*117 'u'*/ {0x00,0x00,0xCC,0xCC,0xCC,0xCC,0x76,0x00},
/*118 'v'*/ {0x00,0x00,0xCC,0xCC,0xCC,0x78,0x30,0x00},
/*119 'w'*/ {0x00,0x00,0xC6,0xD6,0xFE,0xFE,0x6C,0x00},
/*120 'x'*/ {0x00,0x00,0xC6,0x6C,0x38,0x6C,0xC6,0x00},
/*121 'y'*/ {0x00,0x00,0xCC,0xCC,0xCC,0x7C,0x0C,0xF8},
/*122 'z'*/ {0x00,0x00,0xFC,0x98,0x30,0x64,0xFC,0x00},
/*123 '{'*/ {0x1C,0x30,0x30,0xE0,0x30,0x30,0x1C,0x00},
/*124 '|'*/ {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00},
/*125 '}'*/ {0xE0,0x30,0x30,0x1C,0x30,0x30,0xE0,0x00},
/*126 '~'*/ {0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00},
};

// ============================================================================
// 20 Distinct Colors for 80 COCO classes  (R, G, B)
// ============================================================================
static const uint8_t FB_CLASS_COLORS[20][3] = {
    {255,  56,  56}, // 0  red
    { 56, 255,  56}, // 1  green
    { 56,  56, 255}, // 2  blue
    {255, 220,   0}, // 3  yellow
    {  0, 220, 255}, // 4  cyan
    {255,   0, 220}, // 5  magenta
    {255, 140,   0}, // 6  orange
    {160,  32, 240}, // 7  purple
    {  0, 255, 160}, // 8  spring green
    {255,   0, 100}, // 9  rose
    {180, 255,   0}, // 10 chartreuse
    {  0, 100, 255}, // 11 azure
    {255, 180, 180}, // 12 light red
    {180, 255, 180}, // 13 light green
    {180, 180, 255}, // 14 light blue
    {255, 255, 160}, // 15 light yellow
    {160, 255, 255}, // 16 light cyan
    {255, 160, 255}, // 17 light magenta
    {200,  80,   0}, // 18 brown
    {  0, 200,  80}, // 19 teal
};

// ============================================================================
// COCO 80 Class Names
// ============================================================================
static const char* FB_COCO_NAMES[80] = {
    "nguoi","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

// ============================================================================
// FramebufferDisplay
// ============================================================================
class FramebufferDisplay {
public:
    struct Config {
        int  target_width  = 1280;
        int  target_height = 720;
        bool draw_fps  = true;
        bool draw_bbox = true;
    };

    FramebufferDisplay() : fd_(-1), fb_ptr_(nullptr), fb_size_(0),
                           running_(false), frames_displayed_(0),
                           bbox_enabled_(true),
                           servo_enabled_(false), servo_direction_(0),
                           servo_angle_(90.0f) {}
    ~FramebufferDisplay() { stop(); }

    // -------------------------------------------------------------------------
    bool start(const Config& config) {
        config_ = config;

        fd_ = open("/dev/fb0", O_RDWR);
        if (fd_ < 0) {
            perror("Cannot open /dev/fb0 (try: sudo chmod 666 /dev/fb0)");
            return false;
        }

        struct fb_var_screeninfo vinfo;
        struct fb_fix_screeninfo finfo;
        if (ioctl(fd_, FBIOGET_VSCREENINFO, &vinfo) < 0) {
            perror("FBIOGET_VSCREENINFO"); close(fd_); return false;
        }
        if (ioctl(fd_, FBIOGET_FSCREENINFO, &finfo) < 0) {
            perror("FBIOGET_FSCREENINFO"); close(fd_); return false;
        }

        screen_width_   = vinfo.xres;
        screen_height_  = vinfo.yres;
        bits_per_pixel_ = vinfo.bits_per_pixel;
        line_length_    = finfo.line_length;
        fb_size_        = finfo.smem_len;

        fb_ptr_ = static_cast<uint8_t*>(
            mmap(nullptr, fb_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (fb_ptr_ == MAP_FAILED) {
            perror("mmap framebuffer"); close(fd_); return false;
        }

        // Always use full screen
        display_width_  = screen_width_;
        display_height_ = screen_height_;
        offset_x_ = offset_y_ = 0;

        printf("Framebuffer: %dx%d @%dbpp -> FULL SCREEN (labels ON)\n",
               screen_width_, screen_height_, bits_per_pixel_);

        running_ = true;
        return true;
    }

    void stop() {
        running_ = false;
        if (fb_ptr_ && fb_ptr_ != MAP_FAILED) { munmap(fb_ptr_, fb_size_); fb_ptr_ = nullptr; }
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
    }

    // -------------------------------------------------------------------------
    bool push_bgr(const uint8_t* bgr_data, int width, int height, int stride,
                  const DetectionResult& result, float fps = 0, float infer_ms = 0,
                  int cursor_x = -1, int cursor_y = -1) {
        if (!running_ || !fb_ptr_) return false;

        // 1. Blit scaled frame
        if (bits_per_pixel_ == 32)
            blit_scaled_bgra32(bgr_data, width, height, stride);
        else
            blit_scaled_rgb565(bgr_data, width, height, stride);

        // 2. Bounding boxes + labels (controlled by toggle)
        if (bbox_enabled_.load(std::memory_order_relaxed)) {
            for (int i = 0; i < result.count; i++) {
                draw_detection(result.detections[i]);
            }
        }

        // 3. FPS & Inference overlay (top-left)
        if (config_.draw_fps && fps > 0) {
            draw_fps_indicator(fps, infer_ms);
        }

        // 4. Toggle button (bottom-right)
        draw_bbox_toggle_button();

        // 5. Servo buttons (bottom-left, only if enabled)
        if (servo_enabled_) {
            draw_servo_buttons();
        }

        // 6. Mouse cursor (if present)
        if (cursor_x >= 0 && cursor_y >= 0) {
            draw_cursor(cursor_x, cursor_y);
        }

        frames_displayed_++;
        return true;
    }

    // -------------------------------------------------------------------------
    // Input handler — called from TouchInput thread on press AND release
    // -------------------------------------------------------------------------
    void handle_input(int x, int y, bool pressed) {
        if (pressed) {
            // ── PRESS ──
            // Check servo buttons (momentary: active while held)
            if (servo_enabled_) {
                int sx1, sy1, sx2, sy2;
                get_servo_ccw_rect(sx1, sy1, sx2, sy2);
                if (x >= sx1 && x <= sx2 && y >= sy1 && y <= sy2) {
                    servo_direction_.store(-1, std::memory_order_relaxed);
                    if (servo_callback_) servo_callback_(-1);
                    return;
                }
                get_servo_cw_rect(sx1, sy1, sx2, sy2);
                if (x >= sx1 && x <= sx2 && y >= sy1 && y <= sy2) {
                    servo_direction_.store(+1, std::memory_order_relaxed);
                    if (servo_callback_) servo_callback_(+1);
                    return;
                }
            }
        } else {
            // ── RELEASE ──
            // Stop servo on ANY release (even if finger dragged off button)
            if (servo_enabled_ && servo_direction_.load(std::memory_order_relaxed) != 0) {
                servo_direction_.store(0, std::memory_order_relaxed);
                if (servo_callback_) servo_callback_(0);
            }

            // Check BBOX toggle (only activates on release)
            int bx1, by1, bx2, by2;
            get_bbox_button_rect(bx1, by1, bx2, by2);
            if (x >= bx1 && x <= bx2 && y >= by1 && y <= by2) {
                bool cur = bbox_enabled_.load(std::memory_order_relaxed);
                bbox_enabled_.store(!cur, std::memory_order_relaxed);
                printf("BBOX display: %s\n", !cur ? "ON" : "OFF");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Servo configuration
    // -------------------------------------------------------------------------
    using ServoCallback = std::function<void(int direction)>;

    void set_servo_enabled(bool en)         { servo_enabled_ = en; }
    void set_servo_callback(ServoCallback c){ servo_callback_ = std::move(c); }
    void update_servo_angle(float angle)    { servo_angle_ = angle; }
    /// -1 / 0 / +1 while user holds on-screen servo buttons (for merging with remote /servo)
    int servo_touch_direction() const {
        return servo_direction_.load(std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // Public accessors
    // -------------------------------------------------------------------------
    int  frames_displayed() const { return frames_displayed_.load(); }
    bool is_running()       const { return running_.load(); }
    int  screen_width()     const { return screen_width_; }
    int  screen_height()    const { return screen_height_; }
    bool is_bbox_enabled()  const { return bbox_enabled_.load(std::memory_order_relaxed); }
    void set_bbox_enabled(bool v) { bbox_enabled_.store(v, std::memory_order_relaxed); }

private:
    // =========================================================================
    // Pixel primitives
    // =========================================================================

    inline void put_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || y < 0 || x >= display_width_ || y >= display_height_) return;
        if (bits_per_pixel_ == 32) {
            uint8_t* p = fb_ptr_ + (y + offset_y_) * line_length_ + (x + offset_x_) * 4;
            p[0] = b; p[1] = g; p[2] = r; p[3] = 0xFF;
        } else {
            uint16_t* p = reinterpret_cast<uint16_t*>(
                fb_ptr_ + (y + offset_y_) * line_length_) + (x + offset_x_);
            *p = static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
        }
    }

    void draw_hline(int x1, int x2, int y, uint8_t r, uint8_t g, uint8_t b) {
        x1 = std::max(0, x1); x2 = std::min(display_width_ - 1, x2);
        for (int x = x1; x <= x2; x++) put_pixel(x, y, r, g, b);
    }

    void draw_vline(int x, int y1, int y2, uint8_t r, uint8_t g, uint8_t b) {
        y1 = std::max(0, y1); y2 = std::min(display_height_ - 1, y2);
        for (int y = y1; y <= y2; y++) put_pixel(x, y, r, g, b);
    }

    void fill_rect(int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b) {
        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(display_width_  - 1, x2);
        y2 = std::min(display_height_ - 1, y2);
        for (int y = y1; y <= y2; y++) draw_hline(x1, x2, y, r, g, b);
    }

    void draw_rect(int x1, int y1, int x2, int y2,
                   uint8_t r, uint8_t g, uint8_t b, int thick = 2) {
        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(display_width_  - 1, x2);
        y2 = std::min(display_height_ - 1, y2);
        for (int t = 0; t < thick; t++) {
            if (y1 + t <= y2 - t) { draw_hline(x1 + t, x2 - t, y1 + t, r, g, b); draw_hline(x1 + t, x2 - t, y2 - t, r, g, b); }
            if (x1 + t <= x2 - t) { draw_vline(x1 + t, y1 + t, y2 - t, r, g, b); draw_vline(x2 - t, y1 + t, y2 - t, r, g, b); }
        }
    }

    // =========================================================================
    // Font rendering  (scale = 1 → 8×8 px, scale = 2 → 16×16 px)
    // =========================================================================

    void draw_char(int x, int y, char c, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
        int idx = static_cast<unsigned char>(c) - 32;
        if (idx < 0 || idx > 94) idx = 0;  // fallback to space
        const uint8_t* glyph = FONT8x8[idx];
        for (int row = 0; row < 8; row++) {
            uint8_t bits = glyph[row];
            for (int col = 0; col < 8; col++) {
                if (bits & (0x80 >> col)) {
                    for (int sy = 0; sy < scale; sy++)
                        for (int sx = 0; sx < scale; sx++)
                            put_pixel(x + col * scale + sx, y + row * scale + sy, r, g, b);
                }
            }
        }
    }

    // Returns width of rendered text in pixels
    int draw_text(int x, int y, const char* text, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
        int cx = x;
        while (*text) {
            draw_char(cx, y, *text++, r, g, b, scale);
            cx += 8 * scale;
        }
        return cx - x;
    }

    // text width without drawing
    int text_width(const char* text, int scale = 2) {
        int n = 0;
        while (*text++) n++;
        return n * 8 * scale;
    }

    // =========================================================================
    // Detection overlay
    // =========================================================================

    void draw_detection(const Detection& det) {
        // Normalized [0,1] → display pixels
        int x1 = static_cast<int>(det.x1 * display_width_);
        int y1 = static_cast<int>(det.y1 * display_height_);
        int x2 = static_cast<int>(det.x2 * display_width_);
        int y2 = static_cast<int>(det.y2 * display_height_);

        int cid  = std::max(0, std::min(79, det.class_id));
        int cidx = cid % 20;
        uint8_t r = FB_CLASS_COLORS[cidx][0];
        uint8_t g = FB_CLASS_COLORS[cidx][1];
        uint8_t b = FB_CLASS_COLORS[cidx][2];

        // Bounding box (3px thick)
        draw_rect(x1, y1, x2, y2, r, g, b, 3);

        // Build label string: "person 97%"
        const char* cls_name = (cid < 80) ? FB_COCO_NAMES[cid] : "unknown";
        int conf_pct = static_cast<int>(det.confidence * 100.0f + 0.5f);
        char label[48];
        snprintf(label, sizeof(label), "%s %d%%", cls_name, conf_pct);

        // Label background box (above bbox, or inside if no room above)
        int scale    = 2;
        int char_h   = 8 * scale;   // 16
        int pad      = 2;
        int label_w  = text_width(label, scale) + pad * 2;
        int label_h  = char_h + pad * 2;

        int lx1 = x1;
        int ly1 = y1 - label_h;
        int lx2 = x1 + label_w;
        int ly2 = y1 - 1;

        // If above screen, draw inside bbox
        if (ly1 < 0) { ly1 = y1 + 1; ly2 = y1 + label_h; }

        // Black background
        fill_rect(lx1, ly1, lx2, ly2, 0, 0, 0);

        // White text
        draw_text(lx1 + pad, ly1 + pad, label, 255, 255, 255, scale);
    }

    // =========================================================================
    // FPS & Inference overlay (top-left corner)
    // =========================================================================

    void draw_fps_indicator(float fps, float infer_ms) {
        // 1. Build text string
        char text[64];
        snprintf(text, sizeof(text), "FPS: %.1f | Inf: %.1fms", fps, infer_ms);

        // 2. Metrics for rendering box
        int scale  = 2;              // 16x16 pixel per char
        int pad    = 6;              // background padding
        int text_w = text_width(text, scale);
        int text_h = 8 * scale;
        
        int x1 = 10;
        int y1 = 10;
        int x2 = x1 + text_w + pad * 2;
        int y2 = y1 + text_h + pad * 2;

        // 3. Draw black background box for readability
        fill_rect(x1, y1, x2, y2, 0, 0, 0);

        // 4. Choose text color based on performance
        uint8_t r, g, b;
        if      (fps >= 20.0f) { r = 0;   g = 255; b = 0;   } // green  → perfect
        else if (fps >= 10.0f) { r = 255; g = 255; b = 0;   } // yellow → acceptable
        else                   { r = 255; g = 50;  b = 50;  } // red    → slow

        // 5. Draw text inside the box
        draw_text(x1 + pad, y1 + pad, text, r, g, b, scale);
    }

    // =========================================================================
    // BBOX Toggle Button (bottom-right corner)
    // =========================================================================

    /// Compute the button rectangle in screen coordinates
    void get_bbox_button_rect(int& x1, int& y1, int& x2, int& y2) const {
        // Adaptive font scale: smaller font for small displays
        int btn_scale = (display_width_ > 640) ? 2 : 1;
        int char_w    = 8 * btn_scale;
        int char_h    = 8 * btn_scale;
        int pad       = 4 * btn_scale;
        int margin    = 8;

        // "BBOX:ON" = 7 chars, "BBOX:OFF" = 8 chars → use 8 for sizing
        int text_w = 8 * char_w;
        int btn_w  = text_w + pad * 2;
        int btn_h  = char_h + pad * 2;

        x1 = display_width_  - btn_w - margin;
        y1 = display_height_ - btn_h - margin;
        x2 = display_width_  - margin;
        y2 = display_height_ - margin;
    }

    void draw_bbox_toggle_button() {
        int bx1, by1, bx2, by2;
        get_bbox_button_rect(bx1, by1, bx2, by2);

        bool on = bbox_enabled_.load(std::memory_order_relaxed);

        // Button background colour
        uint8_t bg_r, bg_g, bg_b;
        if (on) {
            bg_r = 0; bg_g = 160; bg_b = 0;   // green = ON
        } else {
            bg_r = 160; bg_g = 0; bg_b = 0;    // red   = OFF
        }

        // Background fill
        fill_rect(bx1, by1, bx2, by2, bg_r, bg_g, bg_b);

        // White border (1px)
        draw_rect(bx1, by1, bx2, by2, 255, 255, 255, 1);

        // Label text
        const char* label = on ? "BBOX:ON" : "BBOX:OFF";
        int btn_scale = (display_width_ > 640) ? 2 : 1;
        int pad = 4 * btn_scale;
        draw_text(bx1 + pad, by1 + pad, label, 255, 255, 255, btn_scale);
    }

    // =========================================================================
    // Mouse Cursor (crosshair style)
    // =========================================================================

    void draw_cursor(int cx, int cy) {
        int size = (display_width_ > 640) ? 10 : 6;

        // Horizontal line (black outline + white center)
        for (int dx = -size; dx <= size; dx++) {
            put_pixel(cx + dx, cy - 1, 0, 0, 0);
            put_pixel(cx + dx, cy,     255, 255, 255);
            put_pixel(cx + dx, cy + 1, 0, 0, 0);
        }
        // Vertical line (black outline + white center)
        for (int dy = -size; dy <= size; dy++) {
            put_pixel(cx - 1, cy + dy, 0, 0, 0);
            put_pixel(cx,     cy + dy, 255, 255, 255);
            put_pixel(cx + 1, cy + dy, 0, 0, 0);
        }
        // Center dot (bright red)
        put_pixel(cx, cy, 255, 50, 50);
    }

    // =========================================================================
    // Servo Buttons  (bottom-left:  [<]  090  [>] )
    // =========================================================================

    int servo_btn_scale() const { return (display_width_ > 640) ? 2 : 1; }

    void get_servo_ccw_rect(int& x1, int& y1, int& x2, int& y2) const {
        int s   = servo_btn_scale();
        int cw  = 8 * s;       // char width
        int ch  = 8 * s;       // char height
        int pad = 4 * s;
        int margin = 8;
        // 3 chars: "<" (but padded to look button-like)
        int btn_w = 3 * cw + pad * 2;
        int btn_h = ch + pad * 2;
        x1 = margin;
        y1 = display_height_ - btn_h - margin;
        x2 = x1 + btn_w;
        y2 = y1 + btn_h;
    }

    void get_servo_cw_rect(int& x1, int& y1, int& x2, int& y2) const {
        int s   = servo_btn_scale();
        int cw  = 8 * s;
        int ch  = 8 * s;
        int pad = 4 * s;
        int gap = 4 * s;

        // Position after CCW button + angle text
        int ccw_x1, ccw_y1, ccw_x2, ccw_y2;
        get_servo_ccw_rect(ccw_x1, ccw_y1, ccw_x2, ccw_y2);

        // Angle text width: 3 digits (e.g. "090") = 3 chars
        int angle_w = 3 * cw;

        int btn_w = 3 * cw + pad * 2;
        int btn_h = ch + pad * 2;

        x1 = ccw_x2 + gap + angle_w + gap;
        y1 = ccw_y1;
        x2 = x1 + btn_w;
        y2 = y1 + btn_h;
    }

    void draw_servo_buttons() {
        int s   = servo_btn_scale();
        int pad = 4 * s;
        int gap = 4 * s;
        int dir = servo_direction_.load(std::memory_order_relaxed);

        // ── CCW button ──
        {
            int x1, y1, x2, y2;
            get_servo_ccw_rect(x1, y1, x2, y2);
            uint8_t bg_r = 80, bg_g = 80, bg_b = 80;  // idle: gray
            if (dir == -1) { bg_r = 0; bg_g = 180; bg_b = 0; }  // active: green
            fill_rect(x1, y1, x2, y2, bg_r, bg_g, bg_b);
            draw_rect(x1, y1, x2, y2, 255, 255, 255, 1);
            draw_text(x1 + pad, y1 + pad, " < ", 255, 255, 255, s);
        }

        // ── Angle display (between buttons) ──
        {
            int ccw_x1, ccw_y1, ccw_x2, ccw_y2;
            get_servo_ccw_rect(ccw_x1, ccw_y1, ccw_x2, ccw_y2);

            int angle_x = ccw_x2 + gap;
            int angle_y = ccw_y1 + pad;

            char angle_str[8];
            int angle_int = static_cast<int>(servo_angle_ + 0.5f);
            snprintf(angle_str, sizeof(angle_str), "%03d", std::clamp(angle_int, 0, 180));
            draw_text(angle_x, angle_y, angle_str, 255, 255, 0, s);  // yellow
        }

        // ── CW button ──
        {
            int x1, y1, x2, y2;
            get_servo_cw_rect(x1, y1, x2, y2);
            uint8_t bg_r = 80, bg_g = 80, bg_b = 80;
            if (dir == +1) { bg_r = 0; bg_g = 180; bg_b = 0; }
            fill_rect(x1, y1, x2, y2, bg_r, bg_g, bg_b);
            draw_rect(x1, y1, x2, y2, 255, 255, 255, 1);
            draw_text(x1 + pad, y1 + pad, " > ", 255, 255, 255, s);
        }
    }

    // =========================================================================
    // Scaled blit helpers
    // =========================================================================

    void blit_scaled_bgra32(const uint8_t* bgr, int src_w, int src_h, int src_stride) {
        for (int dy = 0; dy < display_height_; dy++) {
            int sy = dy * src_h / display_height_;
            const uint8_t* src_row = bgr + sy * src_stride;
            uint8_t* dst_row = fb_ptr_ + (dy + offset_y_) * line_length_ + offset_x_ * 4;
            for (int dx = 0; dx < display_width_; dx++) {
                int sx = dx * src_w / display_width_;
                const uint8_t* p = src_row + sx * 3;
                dst_row[dx * 4 + 0] = p[0];
                dst_row[dx * 4 + 1] = p[1];
                dst_row[dx * 4 + 2] = p[2];
                dst_row[dx * 4 + 3] = 0xFF;
            }
        }
    }

    void blit_scaled_rgb565(const uint8_t* bgr, int src_w, int src_h, int src_stride) {
        for (int dy = 0; dy < display_height_; dy++) {
            int sy = dy * src_h / display_height_;
            const uint8_t* src_row = bgr + sy * src_stride;
            uint16_t* dst_row = reinterpret_cast<uint16_t*>(
                fb_ptr_ + (dy + offset_y_) * line_length_) + offset_x_;

            int dx = 0;
            for (; dx + 8 <= display_width_; dx += 8) {
                uint8_t tmp[24];
                for (int k = 0; k < 8; k++) {
                    int sx = (dx + k) * src_w / display_width_;
                    const uint8_t* p = src_row + sx * 3;
                    tmp[k*3+0] = p[0]; tmp[k*3+1] = p[1]; tmp[k*3+2] = p[2];
                }
                uint8x8x3_t px = vld3_u8(tmp);
                uint16x8_t vr = vshrq_n_u16(vmovl_u8(px.val[2]), 3);
                uint16x8_t vg = vshrq_n_u16(vmovl_u8(px.val[1]), 2);
                uint16x8_t vb = vshrq_n_u16(vmovl_u8(px.val[0]), 3);
                uint16x8_t rgb565 = vorrq_u16(vorrq_u16(vshlq_n_u16(vr,11), vshlq_n_u16(vg,5)), vb);
                vst1q_u16(dst_row + dx, rgb565);
            }
            for (; dx < display_width_; dx++) {
                int sx = dx * src_w / display_width_;
                const uint8_t* p = src_row + sx * 3;
                dst_row[dx] = static_cast<uint16_t>(
                    ((p[2] >> 3) << 11) | ((p[1] >> 2) << 5) | (p[0] >> 3));
            }
        }
    }

    // =========================================================================
    // Members
    // =========================================================================
    Config   config_;
    int      fd_;
    uint8_t* fb_ptr_;
    size_t   fb_size_;
    int      screen_width_, screen_height_;
    int      display_width_, display_height_;
    int      bits_per_pixel_, line_length_;
    int      offset_x_, offset_y_;
    std::atomic<bool> running_;
    std::atomic<int>  frames_displayed_;
    std::atomic<bool> bbox_enabled_;          // toggle via touch/click

    // Servo state
    bool              servo_enabled_;
    std::atomic<int>  servo_direction_;       // -1, 0, +1
    float             servo_angle_;           // current angle for display
    ServoCallback     servo_callback_;
};

}  // namespace yolo
#endif  // YOLO_DRM_DISPLAY_H
