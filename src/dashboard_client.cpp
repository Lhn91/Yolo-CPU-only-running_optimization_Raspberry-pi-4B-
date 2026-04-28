#include "dashboard_client.h"

// Never use `#define CPPHTTPLIB_OPENSSL_SUPPORT 0` — #ifdef still enables OpenSSL
// and pulls in EVP_* without -lcrypto. Dashboard only needs plain HTTP.
#include "httplib.h"
#include "json.hpp"
#include <iostream>

namespace yolo {

DashboardClient::DashboardClient() = default;

DashboardClient::~DashboardClient() {
    stop();
}

bool DashboardClient::start(const std::string& base_url) {
    if (running_) return true;

    base_url_ = base_url;
    
    // Simple parsing of http://host:port
    std::string prefix = "http://";
    if (base_url.find(prefix) == 0) {
        std::string host_port = base_url.substr(prefix.length());
        size_t colon_pos = host_port.find(':');
        if (colon_pos != std::string::npos) {
            host_ = host_port.substr(0, colon_pos);
            port_ = std::stoi(host_port.substr(colon_pos + 1));
        } else {
            host_ = host_port;
            port_ = 80;
        }
    } else {
        host_ = base_url;
        port_ = 80;
    }

    running_ = true;
    thread_ = std::thread(&DashboardClient::send_loop, this);
    
    return true;
}

bool DashboardClient::register_stream_url(const std::string& mjpeg_url) {
    if (mjpeg_url.empty() || host_.empty()) {
        return false;
    }
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(2, 0);
    nlohmann::json body = {{"stream_url", mjpeg_url}};
    if (auto res = cli.Post("/stream-config", body.dump(), "application/json")) {
        return res->status == 200;
    }
    return false;
}

void DashboardClient::stop() {
    if (!running_) return;

    running_ = false;
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }
}

void DashboardClient::send_metrics(const std::string& json_data) {
    if (!running_) return;

    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= max_queue_size_) {
        queue_.pop(); // Drop oldest
    }
    queue_.push({DashboardPayload::Type::METRICS, json_data});
    lock.unlock();
    cv_.notify_one();
}

void DashboardClient::send_snapshot(const std::vector<uint8_t>& jpeg_data) {
    if (!running_) return;

    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.size() >= max_queue_size_) {
        queue_.pop(); // Drop oldest
    }
    std::string data(reinterpret_cast<const char*>(jpeg_data.data()), jpeg_data.size());
    queue_.push({DashboardPayload::Type::SNAPSHOT, std::move(data)});
    lock.unlock();
    cv_.notify_one();
}

int DashboardClient::fetch_servo_command() {
    if (host_.empty()) return -999;
    
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(1, 0); // 1 second timeout
    cli.set_read_timeout(1, 0);

    if (auto res = cli.Get("/servo")) {
        if (res->status == 200) {
            // Very simple JSON parsing without heavy dependencies
            // {"direction": 1}
            size_t pos = res->body.find("\"direction\"");
            if (pos != std::string::npos) {
                size_t colon = res->body.find(':', pos);
                if (colon != std::string::npos) {
                    try {
                        return std::stoi(res->body.substr(colon + 1));
                    } catch (...) {
                        return -999;
                    }
                }
            }
        }
    }
    return -999;
}

void DashboardClient::send_loop() {
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(2, 0); // 2 second connection timeout
    cli.set_read_timeout(2, 0); // 2 second read timeout
    cli.set_keep_alive(true); // Keep alive to reduce overhead

    while (running_) {
        DashboardPayload payload;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });

            if (!running_ && queue_.empty()) {
                break;
            }

            payload = std::move(queue_.front());
            queue_.pop();
        }

        if (payload.type == DashboardPayload::Type::METRICS) {
            cli.Post("/metrics", payload.data, "application/json");
        } else if (payload.type == DashboardPayload::Type::SNAPSHOT) {
            httplib::UploadFormDataItems items = {
                { "file", payload.data, "snap.jpg", "image/jpeg" }
            };
            cli.Post("/snapshot", items);
        }
    }
}

} // namespace yolo
