#ifndef DASHBOARD_CLIENT_H
#define DASHBOARD_CLIENT_H

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace yolo {

struct DashboardPayload {
    enum class Type {
        METRICS,
        SNAPSHOT,
        BOXES
    };
    Type type;
    std::string data; // JSON string or JPEG bytes
};

class DashboardClient {
public:
    DashboardClient();
    ~DashboardClient();

    bool start(const std::string& base_url);
    void stop();

    /** POST /stream-config so the web UI shows the MJPEG URL (e.g. Tailscale). */
    bool register_stream_url(const std::string& mjpeg_url);

    void send_metrics(const std::string& json_data);
    void send_snapshot(const std::vector<uint8_t>& jpeg_data);
    
    // Fetch servo command (-1, 0, 1), returns -999 if error
    int fetch_servo_command();

private:
    void send_loop();

    std::string base_url_;
    std::string host_;
    int port_;

    std::atomic<bool> running_{false};
    std::thread thread_;

    std::queue<DashboardPayload> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_queue_size_ = 5;
};

} // namespace yolo

#endif // DASHBOARD_CLIENT_H
