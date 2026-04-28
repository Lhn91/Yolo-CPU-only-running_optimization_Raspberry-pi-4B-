#include "mjpeg_stream_server.h"

#include <cerrno>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>

namespace yolo {

MjpegStreamServer::MjpegStreamServer(int port) : port_(port) {}

MjpegStreamServer::~MjpegStreamServer() { stop(); }

bool MjpegStreamServer::start() {
    if (port_ <= 0 || running_) {
        return false;
    }

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::cerr << "[MJPEG] socket: " << strerror(errno) << "\n";
        return false;
    }

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[MJPEG] bind port " << port_ << ": " << strerror(errno) << "\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::listen(listen_fd_, 8) != 0) {
        std::cerr << "[MJPEG] listen: " << strerror(errno) << "\n";
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    running_ = true;
    thread_ = std::thread(&MjpegStreamServer::accept_thread_main, this);
    std::cout << "[MJPEG] Stream server: http://0.0.0.0:" << port_ << "/stream\n";
    return true;
}

void MjpegStreamServer::stop() {
    running_ = false;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void MjpegStreamServer::update_frame(const std::vector<uint8_t>& jpeg) {
    if (jpeg.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_jpeg_ = jpeg;
}

void MjpegStreamServer::accept_thread_main() {
    while (running_ && listen_fd_ >= 0) {
        sockaddr_in cli{};
        socklen_t clen = sizeof(cli);
        int c = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&cli), &clen);
        if (c < 0) {
            if (running_) {
                std::cerr << "[MJPEG] accept: " << strerror(errno) << "\n";
            }
            break;
        }
        std::thread(&MjpegStreamServer::handle_client, this, c).detach();
    }
}

static bool send_all(int fd, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
        if (w <= 0) {
            return false;
        }
        off += static_cast<size_t>(w);
    }
    return true;
}

void MjpegStreamServer::handle_client(int client_fd) {
    char buf[2048] = {};
    ssize_t n = ::recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        ::close(client_fd);
        return;
    }
    buf[n] = '\0';

    // First line: GET /stream HTTP/1.x
    bool ok_path = (std::strstr(buf, "GET /stream") != nullptr) ||
                   (std::strstr(buf, "GET /stream.mjpg") != nullptr) ||
                   (std::strstr(buf, "GET / ") != nullptr) ||
                   (std::strstr(buf, "GET / HTTP") != nullptr);

    if (!ok_path) {
        const char* resp = "HTTP/1.0 404 Not Found\r\nConnection: close\r\n\r\n";
        (void)send_all(client_fd, resp, std::strlen(resp));
        ::close(client_fd);
        return;
    }

    const char* header =
        "HTTP/1.0 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-store, no-cache, must-revalidate\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    if (!send_all(client_fd, header, std::strlen(header))) {
        ::close(client_fd);
        return;
    }

    const std::string prefix = "--frame\r\nContent-Type: image/jpeg\r\n\r\n";
    const char* suffix = "\r\n";

    while (running_) {
        std::vector<uint8_t> jpg;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            jpg = latest_jpeg_;
        }
        if (jpg.empty()) {
            usleep(20 * 1000);
            continue;
        }
        if (!send_all(client_fd, prefix.data(), prefix.size()) ||
            !send_all(client_fd, jpg.data(), jpg.size()) ||
            !send_all(client_fd, suffix, std::strlen(suffix))) {
            break;
        }
        usleep(33 * 1000); // ~30 fps cap (same order of magnitude as Coral loop)
    }
    ::close(client_fd);
}

} // namespace yolo
