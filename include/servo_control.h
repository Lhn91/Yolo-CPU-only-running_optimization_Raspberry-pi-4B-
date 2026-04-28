/**
 * @file servo_control.h
 * @brief MG996R servo motor control via Linux sysfs hardware PWM
 *
 * Raspberry Pi 4B setup (one-time):
 *   1. Add to /boot/config.txt:   dtoverlay=pwm,pin=12,func=4
 *   2. Reboot
 *   3. Run script will handle permissions automatically
 *
 * Hardware wiring:
 *   - Signal (orange)  → GPIO 12
 *   - VCC    (red)     → External 5V supply (NOT Pi 5V — servo draws up to 2.5A)
 *   - GND    (brown)   → Shared GND with Pi
 */

#ifndef YOLO_SERVO_CONTROL_H
#define YOLO_SERVO_CONTROL_H

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>

namespace yolo {

class ServoController {
public:
    struct Config {
        int   pwm_chip      = 0;       // /sys/class/pwm/pwmchipN
        int   pwm_channel   = 0;       // channel within the chip
        float min_angle     = 0.0f;    // minimum servo angle (degrees)
        float max_angle     = 180.0f;  // maximum servo angle (degrees)
        float initial_angle = 90.0f;   // starting position
        float speed_dps     = 90.0f;   // rotation speed (degrees per second)
        int   min_pulse_us  = 500;     // pulse width at 0°   (microseconds)
        int   max_pulse_us  = 2500;    // pulse width at 180° (microseconds)
        
        bool  is_continuous_360 = true; // Set to true if using a 360-degree continuous servo
        float stop_angle        = 87.0f;// Pulse width that stops the continuous servo
        float continuous_speed  = 15.0f;// Difference from stop_angle (controls speed)
    };

    ServoController()
        : running_(false), direction_(0), current_angle_(90.0f), duty_fd_(-1) {}

    ~ServoController() { stop(); }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    bool start(const Config& config) {
        config_ = config;
        current_angle_.store(config.initial_angle);

        // Build sysfs paths
        char buf[128];
        snprintf(buf, sizeof(buf), "/sys/class/pwm/pwmchip%d", config.pwm_chip);
        chip_path_ = buf;
        snprintf(buf, sizeof(buf), "/sys/class/pwm/pwmchip%d/pwm%d",
                 config.pwm_chip, config.pwm_channel);
        chan_path_ = buf;

        if (!setup_pwm()) return false;

        // Set initial position
        set_duty_ns(angle_to_duty_ns(config.initial_angle));

        running_.store(true);
        move_thread_ = std::thread(&ServoController::move_loop, this);

        printf("ServoController: OK  angle=%.0f°  speed=%.0f°/s\n",
               config.initial_angle, config.speed_dps);
        return true;
    }

    void stop() {
        running_.store(false);
        direction_.store(0);
        if (move_thread_.joinable()) move_thread_.join();

        // Close fast-path fd
        if (duty_fd_ >= 0) { ::close(duty_fd_); duty_fd_ = -1; }

        // Disable & unexport
        write_attr("enable", "0");
        write_path((chip_path_ + "/unexport").c_str(), config_.pwm_channel);
    }

    // -------------------------------------------------------------------------
    // Control  (thread-safe, called from input handler)
    // -------------------------------------------------------------------------

    /// Set rotation direction: -1 = CCW, 0 = stop, +1 = CW
    void set_direction(int dir) {
        direction_.store(std::clamp(dir, -1, 1), std::memory_order_relaxed);
    }

    /// Current angle in degrees
    float get_angle() const {
        return current_angle_.load(std::memory_order_relaxed);
    }

    int get_direction() const {
        return direction_.load(std::memory_order_relaxed);
    }

private:
    Config      config_;
    std::string chip_path_;   // /sys/class/pwm/pwmchipN
    std::string chan_path_;   // /sys/class/pwm/pwmchipN/pwmM

    std::atomic<bool>  running_;
    std::atomic<int>   direction_;
    std::atomic<float> current_angle_;
    std::thread        move_thread_;
    int                duty_fd_;   // kept open for fast writes

    // =========================================================================
    // PWM setup via sysfs
    // =========================================================================

    bool setup_pwm() {
        // Export channel
        if (!write_path((chip_path_ + "/export").c_str(), config_.pwm_channel)) {
            fprintf(stderr,
                "ServoController: Cannot access %s\n"
                "  ► Add to /boot/config.txt:\n"
                "      dtoverlay=pwm,pin=12,func=4\n"
                "  ► Then reboot.\n", chip_path_.c_str());
            return false;
        }

        // Wait for sysfs to create channel directory
        for (int i = 0; i < 20; i++) {
            if (access((chan_path_ + "/period").c_str(), W_OK) == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // 50 Hz = 20 ms period
        if (!write_attr("period", "20000000")) {
            fprintf(stderr, "ServoController: Cannot set period\n");
            return false;
        }

        // Initial duty cycle
        char duty_str[16];
        snprintf(duty_str, sizeof(duty_str), "%d",
                 angle_to_duty_ns(config_.initial_angle));
        if (!write_attr("duty_cycle", duty_str)) {
            fprintf(stderr, "ServoController: Cannot set duty_cycle\n");
            return false;
        }

        // Open duty_cycle file for fast repeated writes
        std::string duty_path = chan_path_ + "/duty_cycle";
        duty_fd_ = ::open(duty_path.c_str(), O_WRONLY);

        // Enable output
        if (!write_attr("enable", "1")) {
            fprintf(stderr, "ServoController: Cannot enable PWM\n");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Angle ↔ PWM conversion
    // =========================================================================

    int angle_to_duty_ns(float angle) const {
        float frac = (std::clamp(angle, config_.min_angle, config_.max_angle)
                      - config_.min_angle)
                     / (config_.max_angle - config_.min_angle);
        int pulse_us = config_.min_pulse_us
                     + static_cast<int>(frac * (config_.max_pulse_us - config_.min_pulse_us));
        return pulse_us * 1000;   // µs → ns
    }

    /// Fast duty-cycle write (no fopen/fclose overhead)
    void set_duty_ns(int ns) {
        if (duty_fd_ < 0) return;
        char buf[16];
        int len = snprintf(buf, sizeof(buf), "%d", ns);
        lseek(duty_fd_, 0, SEEK_SET);
        (void)::write(duty_fd_, buf, len);
    }

    // =========================================================================
    // Movement thread  (runs at ~50 Hz)
    // =========================================================================

    void move_loop() {
        using Clock = std::chrono::steady_clock;
        auto prev = Clock::now();

        while (running_.load(std::memory_order_relaxed)) {
            auto now = Clock::now();
            float dt = std::chrono::duration<float>(now - prev).count();
            prev = now;

            int dir = direction_.load(std::memory_order_relaxed);
            
            if (config_.is_continuous_360) {
                // Continuous 360 servo mode:
                // PWM pulse controls SPEED and DIRECTION, not position.
                // 87 degrees (~1466us) is STOP.
                // > 87 is rotate one way, < 87 is rotate the other way.
                float target_angle = config_.stop_angle;
                if (dir == 1) {
                    target_angle = config_.stop_angle + config_.continuous_speed;
                } else if (dir == -1) {
                    target_angle = config_.stop_angle - config_.continuous_speed;
                }
                
                current_angle_.store(target_angle, std::memory_order_relaxed);
                set_duty_ns(angle_to_duty_ns(target_angle));
            } else {
                // Positional 180 servo mode:
                // Update angle over time based on direction.
                if (dir != 0) {
                    float angle = current_angle_.load(std::memory_order_relaxed);
                    angle += dir * config_.speed_dps * dt;
                    angle = std::clamp(angle, config_.min_angle, config_.max_angle);
                    current_angle_.store(angle, std::memory_order_relaxed);
                    set_duty_ns(angle_to_duty_ns(angle));
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // =========================================================================
    // sysfs helpers
    // =========================================================================

    bool write_attr(const char* attr, const char* value) {
        std::string path = chan_path_ + "/" + attr;
        FILE* f = fopen(path.c_str(), "w");
        if (!f) return false;
        fputs(value, f);
        fclose(f);
        return true;
    }

    static bool write_path(const char* path, int value) {
        FILE* f = fopen(path, "w");
        if (!f) return false;
        fprintf(f, "%d", value);
        fclose(f);
        return true;
    }
};

}  // namespace yolo
#endif  // YOLO_SERVO_CONTROL_H
