#ifndef MJPEG_STREAM_SERVER_H
#define MJPEG_STREAM_SERVER_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace yolo {

/**
 * Minimal HTTP MJPEG server (same idea as CoralVisionRT main.py), bound to 0.0.0.0.
 * update_frame() is called from the inference thread with the latest JPEG bytes.
 */
class MjpegStreamServer {
public:
    explicit MjpegStreamServer(int port);
    ~MjpegStreamServer();

    MjpegStreamServer(const MjpegStreamServer&) = delete;
    MjpegStreamServer& operator=(const MjpegStreamServer&) = delete;

    bool start();
    void stop();

    void update_frame(const std::vector<uint8_t>& jpeg);

    int port() const { return port_; }

private:
    void accept_thread_main();
    void handle_client(int client_fd);

    int port_ = 0;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;

    std::mutex frame_mutex_;
    std::vector<uint8_t> latest_jpeg_;
};

} // namespace yolo

#endif
