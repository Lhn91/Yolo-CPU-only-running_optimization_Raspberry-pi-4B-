/**
 * @file touch_input.h
 * @brief Linux evdev input handler for touchscreen & mouse (no X11 required)
 *
 * Supports:
 *  - ADS7846 / XPT2046 resistive touchscreen (absolute coordinates)
 *  - USB mouse (relative coordinates, cursor tracking)
 *  - Background polling thread with low CPU usage
 *  - Callback on touch-release / click for virtual button hit-testing
 *
 * USAGE:
 *   sudo chmod 666 /dev/input/event*   (or add user to 'input' group)
 */

#ifndef YOLO_TOUCH_INPUT_H
#define YOLO_TOUCH_INPUT_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <linux/input.h>
#include <sys/ioctl.h>

namespace yolo {

// ============================================================================
// TouchInput — reads raw evdev events, maps to screen coordinates
// ============================================================================

class TouchInput {
public:
    /// Called on touch press/release or mouse click press/release
    /// pressed=true on finger-down / button-down, false on up
    using TouchCallback = std::function<void(int x, int y, bool pressed)>;

    TouchInput()
        : running_(false), screen_w_(0), screen_h_(0),
          cursor_x_(-1), cursor_y_(-1), has_mouse_(false) {}

    ~TouchInput() { stop(); }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    bool start(int screen_width, int screen_height) {
        screen_w_ = screen_width;
        screen_h_ = screen_height;
        cursor_x_.store(screen_width / 2);
        cursor_y_.store(screen_height / 2);

        if (!open_input_devices()) {
            fprintf(stderr, "TouchInput: No touchscreen or mouse found\n");
            return false;
        }

        running_.store(true);
        poll_thread_ = std::thread(&TouchInput::poll_loop, this);
        return true;
    }

    void stop() {
        running_.store(false);
        if (poll_thread_.joinable()) {
            poll_thread_.join();
        }
        for (auto& dev : devices_) {
            if (dev.fd >= 0) { ::close(dev.fd); dev.fd = -1; }
        }
        devices_.clear();
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Register callback fired on touch-release / mouse-click-release
    void set_touch_callback(TouchCallback cb) { touch_callback_ = std::move(cb); }

    // -------------------------------------------------------------------------
    // Cursor state (thread-safe reads)
    // -------------------------------------------------------------------------

    /// Get current mouse cursor position. Returns false if no mouse detected.
    bool get_cursor(int& x, int& y) const {
        if (!has_mouse_) return false;
        x = cursor_x_.load(std::memory_order_relaxed);
        y = cursor_y_.load(std::memory_order_relaxed);
        return true;
    }

    bool has_mouse() const { return has_mouse_; }

private:
    // =========================================================================
    // Input device descriptor
    // =========================================================================
    struct InputDevice {
        int         fd = -1;
        std::string path;
        std::string name;
        bool        is_touchscreen = false;  // ABS events
        bool        is_mouse       = false;  // REL events

        // Touchscreen calibration (from EVIOCGABS)
        int abs_x_min = 0, abs_x_max = 4095;
        int abs_y_min = 0, abs_y_max = 4095;

        // Current touch state
        int  touch_x = 0, touch_y = 0;
        bool touching = false;

        // Mouse cursor state (per-device, merged into global)
        int cursor_x = 0, cursor_y = 0;
    };

    // =========================================================================
    // Device discovery
    // =========================================================================
    bool open_input_devices() {
        for (int i = 0; i < 20; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/dev/input/event%d", i);

            int fd = ::open(path, O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;

            // Query device name
            char name[256] = "Unknown";
            ioctl(fd, EVIOCGNAME(sizeof(name)), name);

            // Query event type capabilities
            uint8_t evbits[(EV_MAX + 7) / 8] = {};
            ioctl(fd, EVIOCGBIT(0, sizeof(evbits)), evbits);

            bool has_abs = (evbits[EV_ABS / 8] >> (EV_ABS % 8)) & 1;
            bool has_rel = (evbits[EV_REL / 8] >> (EV_REL % 8)) & 1;

            // Query key capabilities (BTN_TOUCH for touchscreen, BTN_LEFT for mouse)
            uint8_t keybits[(KEY_MAX + 7) / 8] = {};
            ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keybits)), keybits);

            bool has_btn_touch = (keybits[BTN_TOUCH / 8] >> (BTN_TOUCH % 8)) & 1;
            bool has_btn_left  = (keybits[BTN_LEFT  / 8] >> (BTN_LEFT  % 8)) & 1;

            InputDevice dev;
            dev.fd   = fd;
            dev.path = path;
            dev.name = name;
            dev.cursor_x = screen_w_ / 2;
            dev.cursor_y = screen_h_ / 2;

            // Identify touchscreen: ABS axes + BTN_TOUCH
            if (has_abs && has_btn_touch) {
                struct input_absinfo abs_x, abs_y;
                if (ioctl(fd, EVIOCGABS(ABS_X), &abs_x) == 0 &&
                    ioctl(fd, EVIOCGABS(ABS_Y), &abs_y) == 0) {
                    dev.is_touchscreen = true;
                    dev.abs_x_min = abs_x.minimum;
                    dev.abs_x_max = abs_x.maximum;
                    dev.abs_y_min = abs_y.minimum;
                    dev.abs_y_max = abs_y.maximum;
                    printf("TouchInput: Touchscreen '%s' @ %s  X[%d..%d] Y[%d..%d]\n",
                           name, path,
                           abs_x.minimum, abs_x.maximum,
                           abs_y.minimum, abs_y.maximum);
                }
            }

            // Identify mouse: REL axes + BTN_LEFT (exclude touchscreens that also report REL)
            if (has_rel && has_btn_left && !dev.is_touchscreen) {
                dev.is_mouse = true;
                has_mouse_ = true;
                printf("TouchInput: Mouse '%s' @ %s\n", name, path);
            }

            if (dev.is_touchscreen || dev.is_mouse) {
                devices_.push_back(dev);
            } else {
                ::close(fd);
            }
        }

        return !devices_.empty();
    }

    // =========================================================================
    // Event polling (background thread)
    // =========================================================================
    void poll_loop() {
        std::vector<struct pollfd> pfds;
        pfds.reserve(devices_.size());
        for (auto& dev : devices_) {
            struct pollfd pfd{};
            pfd.fd     = dev.fd;
            pfd.events = POLLIN;
            pfds.push_back(pfd);
        }

        while (running_.load(std::memory_order_relaxed)) {
            int ret = poll(pfds.data(), pfds.size(), 50);  // 50ms timeout → ~20 Hz when idle
            if (ret <= 0) continue;

            for (size_t i = 0; i < pfds.size(); i++) {
                if (pfds[i].revents & POLLIN) {
                    process_events(devices_[i]);
                }
            }
        }
    }

    void process_events(InputDevice& dev) {
        struct input_event ev;
        while (::read(dev.fd, &ev, sizeof(ev)) == static_cast<ssize_t>(sizeof(ev))) {
            if (dev.is_touchscreen) handle_touch(dev, ev);
            if (dev.is_mouse)       handle_mouse(dev, ev);
        }
    }

    // =========================================================================
    // Touchscreen events  (ABS_X, ABS_Y, BTN_TOUCH)
    // =========================================================================
    void handle_touch(InputDevice& dev, const struct input_event& ev) {
        if (ev.type == EV_ABS) {
            if (ev.code == ABS_X) {
                dev.touch_x = map_value(ev.value,
                                        dev.abs_x_min, dev.abs_x_max,
                                        0, screen_w_ - 1);
            } else if (ev.code == ABS_Y) {
                dev.touch_y = map_value(ev.value,
                                        dev.abs_y_min, dev.abs_y_max,
                                        0, screen_h_ - 1);
            }
        } else if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
            if (ev.value == 1) {
                dev.touching = true;
                fire_input(dev.touch_x, dev.touch_y, true);   // PRESS
            } else if (ev.value == 0 && dev.touching) {
                dev.touching = false;
                fire_input(dev.touch_x, dev.touch_y, false);  // RELEASE
            }
        }
    }

    // =========================================================================
    // Mouse events  (REL_X, REL_Y, BTN_LEFT)
    // =========================================================================
    void handle_mouse(InputDevice& dev, const struct input_event& ev) {
        if (ev.type == EV_REL) {
            if (ev.code == REL_X) {
                dev.cursor_x = std::clamp(dev.cursor_x + ev.value, 0, screen_w_ - 1);
                cursor_x_.store(dev.cursor_x, std::memory_order_relaxed);
            } else if (ev.code == REL_Y) {
                dev.cursor_y = std::clamp(dev.cursor_y + ev.value, 0, screen_h_ - 1);
                cursor_y_.store(dev.cursor_y, std::memory_order_relaxed);
            }
        } else if (ev.type == EV_KEY && ev.code == BTN_LEFT) {
            if (ev.value == 1) {
                fire_input(dev.cursor_x, dev.cursor_y, true);   // PRESS
            } else if (ev.value == 0) {
                fire_input(dev.cursor_x, dev.cursor_y, false);  // RELEASE
            }
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    static int map_value(int v, int in_min, int in_max, int out_min, int out_max) {
        if (in_max == in_min) return out_min;
        int64_t numer = static_cast<int64_t>(v - in_min) * (out_max - out_min);
        return out_min + static_cast<int>(numer / (in_max - in_min));
    }

    void fire_input(int x, int y, bool pressed) {
        if (touch_callback_) {
            touch_callback_(x, y, pressed);
        }
    }

    // =========================================================================
    // Members
    // =========================================================================
    std::atomic<bool> running_;
    int screen_w_, screen_h_;

    std::vector<InputDevice> devices_;
    std::thread              poll_thread_;

    TouchCallback touch_callback_;

    // Global cursor position (updated by any mouse device)
    std::atomic<int> cursor_x_;
    std::atomic<int> cursor_y_;
    bool             has_mouse_;
};

}  // namespace yolo
#endif  // YOLO_TOUCH_INPUT_H
