/**
 * @file main.cpp
 * @brief Entry point for YOLOv8n realtime inference system - OPTIMIZED
 * 
 * RADICAL OPTIMIZATIONS:
 * - Direct BGR path for video (no YUYV conversion)
 * - FP32 throughout (no FP16 overhead)
 * - Pre-allocated buffers
 * - Optimized thread configuration
 */

#include "common.h"
#include "input_pipeline.h"
#include "neon_preprocess.h"
#include "inference_engine.h"
#include "postprocess.h"
#include "benchmark.h"
#include "video_writer.h"
#include "drm_display.h"
#include "touch_input.h"
#include "servo_control.h"
#include "dashboard_client.h"
#include "mjpeg_stream_server.h"
#include "json.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <signal.h>
#include <getopt.h>
#include <set>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <cmath>
#include <limits>
#include <system_error>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// #region agent log
#warning "DEBUG_SYNC_a1fd6f_main_cpp"
// #endregion

using namespace yolo;

// ============================================================================
// Global State
// ============================================================================

std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    g_running.store(false);
}

// ============================================================================
// Command Line Options
// ============================================================================

struct Options {
    std::string mode = "benchmark";      // benchmark, camera, video
    std::string device = "/dev/video0";  // Camera device or video file
    std::string param_path;              // Model param file
    std::string bin_path;                // Model bin file
    std::string images_dir;             // Folder of images (runs image inference)
    std::string preds_dir;              // Folder to export YOLO predictions (for mAP eval)
    int frames = 1000;                   // Frames to process
    int warmup_frames = 30;              // Warmup frames
    int duration_sec = 0;                // Wall-clock duration after warmup
    int snapshot_sec = 0;                // Snapshot interval for time-window metrics
    bool verbose = false;                // Verbose output
    bool test_model = false;             // Test model loading only
    bool test_inference = false;         // Test single inference only
    bool test_camera = false;            // Test camera capture only
    std::string output_csv;              // Output CSV path
    std::string interval_output_csv;     // Interval metrics CSV path
    std::string output_image_detail_csv;   // Per-image CSV: item,time_ms,fps,detections
    std::string output_image_summary_csv;  // Summary CSV: metric,value
    std::string output_video;            // Output video path (with bbox)
    bool show_fps = true;                // Show FPS overlay in output video
    std::set<int> class_filter;          // Classes to detect (empty = all)
    std::string class_filter_str;        // Original class names string
    bool display_enabled = false;        // Enable display window (OpenCV)
    bool fb_display_enabled = false;      // Enable framebuffer display (no X11)
    bool use_vulkan = false;             // Use Vulkan GPU compute
    bool use_int8 = false;               // Use INT8 quantized model
    int gpu_device = 0;                  // Vulkan GPU device index
    bool servo_enabled = false;          // Enable servo control via PWM
    std::string dashboard_url;           // Dashboard URL (e.g. http://127.0.0.1:5000)
    std::string dashboard_stream_url;    // Optional MJPEG URL for /stream-config (manual override)
    int mjpeg_port = 0;                  // 0 = off; else serve integrated MJPEG on 0.0.0.0:port (like Coral)
    int mjpeg_quality = 72;             // JPEG quality for dashboard + MJPEG stream
    std::string tailscale_ip;           // Optional 100.x override for stream URL registration
};

void print_usage(const char* program) {
    std::cout << "YOLOv8n Realtime Inference System for Raspberry Pi 5\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --benchmark          Run benchmark with video or synthetic data\n";
    std::cout << "  --camera DEVICE      Run with camera input\n";
    std::cout << "  --video FILE         Run with video file input\n\n";
    std::cout << "  --images-dir DIR    Run inference on image folder (YOLO folder-style evaluation)\n\n";
    std::cout << "Model:\n";
    std::cout << "  --param FILE         Path to NCNN .param file\n";
    std::cout << "  --bin FILE           Path to NCNN .bin file\n\n";
    std::cout << "Options:\n";
    std::cout << "  --frames N           Number of frames to process (default: 1000)\n";
    std::cout << "  --warmup N           Warmup frames (default: 30)\n";
    std::cout << "  --output FILE        Export results to CSV\n";
    std::cout << "  --duration-sec N     Run measured benchmark for N seconds after warmup\n";
    std::cout << "  --snapshot-sec N     Write interval metrics every N seconds\n";
    std::cout << "  --interval-output FILE Export interval metrics CSV\n";
    std::cout << "  --output-detail FILE Export per-image detail CSV\n";
    std::cout << "  --output-summary FILE Export per-image summary CSV\n";
    std::cout << "  --preds-dir DIR      Export predictions (YOLO txt) for mAP eval\n";
    std::cout << "  --output-video FILE  Save video with bounding boxes\n";
    std::cout << "  --no-fps             Don't show FPS overlay in output video\n";
    std::cout << "  --class NAMES        Filter classes (comma-separated, e.g., 'person,car,dog')\n";
    std::cout << "  --display            Show detection results in window (auto DISPLAY=:0)\n";
    std::cout << "  --vulkan             Use Vulkan GPU (VideoCore VII) for inference\n";
    std::cout << "  --int8               Use INT8 quantized model (faster, similar accuracy)\n";
    std::cout << "  --servo              Enable servo motor control (GPIO 12 PWM)\n";
    std::cout << "  --dashboard URL      Send metrics and video to Dashboard URL (e.g. http://127.0.0.1:5000)\n";
    std::cout << "  --mjpeg-port N       Serve MJPEG on 0.0.0.0:N/stream (same process; use with --dashboard, default off)\n";
    std::cout << "  --mjpeg-quality N    JPEG quality 30-95 for stream/snapshot (default 72)\n";
    std::cout << "  --tailscale-ip ADDR  Pi 100.x.x.x for registered stream URL (else: tailscale CLI or LAN IP)\n";
    std::cout << "  --dashboard-stream-url URL  Force this /stream-config URL (skip auto if set)\n";
    std::cout << "  --gpu N              Vulkan GPU device index (default: 0)\n";
    std::cout << "  --verbose            Print per-frame results\n\n";
    std::cout << "Testing:\n";
    std::cout << "  --test-model         Test model loading\n";
    std::cout << "  --test-inference     Test single frame inference\n";
    std::cout << "  --test-camera        Test camera capture\n\n";
}

bool parse_options(int argc, char* argv[], Options& opts) {
    static struct option long_options[] = {
        {"benchmark", no_argument, 0, 'b'},
        {"camera", required_argument, 0, 'c'},
        {"video", required_argument, 0, 'v'},
        {"images-dir", required_argument, 0, 'x'},
        {"param", required_argument, 0, 'p'},
        {"bin", required_argument, 0, 'm'},
        {"frames", required_argument, 0, 'n'},
        {"warmup", required_argument, 0, 'w'},
        {"output", required_argument, 0, 'o'},
        {"duration-sec", required_argument, 0, 'u'},
        {"snapshot-sec", required_argument, 0, 'q'},
        {"interval-output", required_argument, 0, 'l'},
        {"output-detail", required_argument, 0, 'a'},
        {"output-summary", required_argument, 0, 's'},
        {"output-video", required_argument, 0, 'O'},
        {"preds-dir", required_argument, 0, 'r'},
        {"no-fps", no_argument, 0, 'F'},
        {"class", required_argument, 0, 'C'},
        {"display", no_argument, 0, 'D'},
        {"fb", no_argument, 0, 'B'},
        {"vulkan", no_argument, 0, 'G'},
        {"int8", no_argument, 0, 'I'},
        {"gpu", required_argument, 0, 'g'},
        {"verbose", no_argument, 0, 'V'},
        {"test-model", no_argument, 0, '1'},
        {"test-inference", no_argument, 0, '2'},
        {"test-camera", no_argument, 0, '3'},
        {"device", required_argument, 0, 'd'},
        {"servo", no_argument, 0, 'Z'},
        {"dashboard", required_argument, 0, 'U'},
        {"dashboard-stream-url", required_argument, 0, 'T'},
        {"mjpeg-port", required_argument, 0, 'J'},
        {"mjpeg-quality", required_argument, 0, 'K'},
        {"tailscale-ip", required_argument, 0, 'Y'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "bc:v:x:p:m:n:w:o:u:q:l:a:s:O:FC:DBGIg:Vd:r:hZUTJK:Y:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                opts.mode = "benchmark";
                break;
            case 'c':
                opts.mode = "camera";
                opts.device = optarg;
                break;
            case 'v':
                opts.mode = "video";
                opts.device = optarg;
                break;
            case 'x':
                opts.mode = "images";
                opts.images_dir = optarg;
                break;
            case 'p':
                opts.param_path = optarg;
                break;
            case 'm':
                opts.bin_path = optarg;
                break;
            case 'n':
                opts.frames = std::stoi(optarg);
                break;
            case 'w':
                opts.warmup_frames = std::stoi(optarg);
                break;
            case 'o':
                opts.output_csv = optarg;
                break;
            case 'u':
                opts.duration_sec = std::stoi(optarg);
                break;
            case 'q':
                opts.snapshot_sec = std::stoi(optarg);
                break;
            case 'l':
                opts.interval_output_csv = optarg;
                break;
            case 'a':
                opts.output_image_detail_csv = optarg;
                break;
            case 's':
                opts.output_image_summary_csv = optarg;
                break;
            case 'O':
                opts.output_video = optarg;
                break;
            case 'r':
                opts.preds_dir = optarg;
                break;
            case 'F':
                opts.show_fps = false;
                break;
            case 'C':
                opts.class_filter_str = optarg;
                break;
            case 'D':
                opts.display_enabled = true;
                break;
            case 'B':
                opts.fb_display_enabled = true;
                break;
            case 'G':
                opts.use_vulkan = true;
                break;
            case 'I':
                opts.use_int8 = true;
                break;
            case 'g':
                opts.gpu_device = std::stoi(optarg);
                break;
            case 'V':
                opts.verbose = true;
                break;
            case 'd':
                opts.device = optarg;
                break;
            case 'Z':
                opts.servo_enabled = true;
                break;
            case 'U':
                opts.dashboard_url = optarg;
                break;
            case 'T':
                opts.dashboard_stream_url = optarg;
                break;
            case 'J':
                opts.mjpeg_port = std::stoi(optarg);
                break;
            case 'K':
                opts.mjpeg_quality = std::stoi(optarg);
                break;
            case 'Y':
                opts.tailscale_ip = optarg;
                break;
            case '1':
                opts.test_model = true;
                break;
            case '2':
                opts.test_inference = true;
                break;
            case '3':
                opts.test_camera = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return false;
            default:
                print_usage(argv[0]);
                return false;
        }
    }

    // Validate required options
    if (!opts.test_camera && (opts.param_path.empty() || opts.bin_path.empty())) {
        std::cerr << "Error: --param and --bin are required\n";
        return false;
    }

    if (opts.duration_sec < 0 || opts.snapshot_sec < 0) {
        std::cerr << "Error: --duration-sec and --snapshot-sec must be >= 0\n";
        return false;
    }

    if (opts.mjpeg_quality < 30) {
        opts.mjpeg_quality = 30;
    }
    if (opts.mjpeg_quality > 95) {
        opts.mjpeg_quality = 95;
    }
    if (opts.mjpeg_port < 0) {
        std::cerr << "Error: --mjpeg-port must be >= 0\n";
        return false;
    }

    if (!opts.interval_output_csv.empty() && opts.snapshot_sec == 0) {
        opts.snapshot_sec = 900;
    }

    if (opts.snapshot_sec > 0 && opts.interval_output_csv.empty()) {
        opts.interval_output_csv = "interval_metrics.csv";
    }

    // Parse class filter if specified
    if (!opts.class_filter_str.empty()) {
        // COCO class names (must match video_writer.h)
        static const char* COCO_NAMES[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };
        constexpr int NUM_COCO_CLASSES = 80;
        
        std::stringstream ss(opts.class_filter_str);
        std::string class_name;
        while (std::getline(ss, class_name, ',')) {
            // Trim whitespace
            class_name.erase(0, class_name.find_first_not_of(" \t"));
            class_name.erase(class_name.find_last_not_of(" \t") + 1);
            
            // Convert to lowercase for comparison
            std::string class_lower = class_name;
            std::transform(class_lower.begin(), class_lower.end(), class_lower.begin(), ::tolower);
            
            bool found = false;
            for (int i = 0; i < NUM_COCO_CLASSES; i++) {
                std::string coco_lower = COCO_NAMES[i];
                std::transform(coco_lower.begin(), coco_lower.end(), coco_lower.begin(), ::tolower);
                if (class_lower == coco_lower) {
                    opts.class_filter.insert(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Warning: Unknown class '" << class_name << "' ignored\n";
                std::cerr << "Available classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, etc.\n";
            }
        }
        
        if (!opts.class_filter.empty()) {
            std::cout << "Class filter: ";
            for (int id : opts.class_filter) {
                std::cout << COCO_NAMES[id] << " ";
            }
            std::cout << "(" << opts.class_filter.size() << " classes)\n";
        }
    }

    return true;
}

static std::string make_timestamp_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf = {};

#ifdef _WIN32
    localtime_s(&tm_buf, &now_time);
#else
    localtime_r(&now_time, &tm_buf);
#endif

    char buffer[32];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &tm_buf) == 0) {
        return "";
    }
    return buffer;
}

static double read_system_temperature_celsius() {
    {
        std::ifstream sysfs("/sys/class/thermal/thermal_zone0/temp");
        double milli_celsius = 0.0;
        if (sysfs >> milli_celsius) {
            return milli_celsius / 1000.0;
        }
    }

    FILE* pipe = popen("vcgencmd measure_temp 2>/dev/null", "r");
    if (!pipe) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    char buffer[128] = {};
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        return std::numeric_limits<double>::quiet_NaN();
    }

    pclose(pipe);

    double temperature_c = 0.0;
    if (sscanf(buffer, "temp=%lf", &temperature_c) == 1) {
        return temperature_c;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

static std::string detect_tailscale_ip_v4() {
    FILE* pipe = popen("tailscale ip -4 2>/dev/null", "r");
    if (!pipe) {
        return "";
    }
    char buf[96] = {};
    if (!fgets(buf, sizeof(buf), pipe)) {
        pclose(pipe);
        return "";
    }
    pclose(pipe);
    std::string s = buf;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

static std::string detect_local_ip_udp() {
    int s = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) {
        return "";
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(53);
    if (inet_pton(AF_INET, "8.8.8.8", &addr.sin_addr) != 1) {
        ::close(s);
        return "";
    }
    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(s);
        return "";
    }
    sockaddr_in name{};
    socklen_t len = sizeof(name);
    if (::getsockname(s, reinterpret_cast<sockaddr*>(&name), &len) != 0) {
        ::close(s);
        return "";
    }
    char out[INET_ADDRSTRLEN] = {};
    if (!inet_ntop(AF_INET, &name.sin_addr, out, sizeof(out))) {
        ::close(s);
        return "";
    }
    ::close(s);
    return std::string(out);
}

static bool ensure_parent_directory_exists(const std::string& path) {
    namespace fs = std::filesystem;

    fs::path out_path(path);
    fs::path parent = out_path.parent_path();
    if (parent.empty()) {
        return true;
    }

    std::error_code ec;
    fs::create_directories(parent, ec);
    return !ec;
}

static bool initialize_interval_metrics_csv(const std::string& path) {
    if (path.empty()) return false;

    if (!ensure_parent_directory_exists(path)) {
        std::cerr << "Warning: cannot create interval output directory for " << path << "\n";
        return false;
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: cannot open interval csv for writing: " << path << "\n";
        return false;
    }

    file << "timestamp,elapsed_sec,window_start_sec,window_end_sec,frames_window,detections_mean,"
         << "latency_mean_us,latency_p50_us,latency_p95_us,latency_p99_us,"
         << "fps_mean,fps_p99,frames_over_50ms,peak_memory_mb,temperature_c\n";
    return true;
}

static bool append_interval_metrics_csv(
    const std::string& path,
    double elapsed_sec,
    double window_start_sec,
    double window_end_sec,
    const BenchmarkStats& stats,
    double temperature_c
) {
    if (path.empty()) return false;

    std::ofstream file(path, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Warning: cannot append interval csv: " << path << "\n";
        return false;
    }

    file << make_timestamp_string() << ","
         << std::fixed << std::setprecision(3) << elapsed_sec << ","
         << window_start_sec << ","
         << window_end_sec << ","
         << stats.total_frames << ","
         << std::setprecision(4) << stats.mean_detections << ","
         << std::setprecision(1) << stats.mean_total_us << ","
         << stats.p50_total_us << ","
         << stats.p95_total_us << ","
         << stats.p99_total_us << ","
         << std::setprecision(4) << stats.fps_mean << ","
         << stats.fps_p99 << ","
         << stats.frames_over_50ms << ","
         << std::setprecision(4) << (static_cast<double>(stats.peak_memory_kb) / 1024.0) << ",";

    if (!std::isnan(temperature_c)) {
        file << std::setprecision(3) << temperature_c;
    }

    file << "\n";
    return true;
}

static void write_image_detail_csv(
    const std::string& path,
    const std::vector<std::string>& items,
    const std::vector<double>& time_ms,
    const std::vector<double>& fps,
    const std::vector<int>& detections
) {
    if (path.empty()) return;
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: cannot open detail csv for writing: " << path << "\n";
        return;
    }

    file << "item,time_ms,fps,detections\n";
    for (size_t i = 0; i < items.size(); i++) {
        file << items[i] << ","
             << std::fixed << std::setprecision(4) << time_ms[i] << ","
             << std::fixed << std::setprecision(4) << fps[i] << ","
             << detections[i] << "\n";
    }
    file.close();
}

static void write_image_summary_csv(
    const std::string& path,
    const std::string& model_name,
    const std::string& images_dir,
    size_t samples,
    double total_time_s,
    double fps_overall,
    double time_ms_mean,
    double time_ms_std,
    double fps_mean,
    double fps_min,
    double fps_max,
    double detections_mean
) {
    if (path.empty()) return;
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: cannot open summary csv for writing: " << path << "\n";
        return;
    }

    file << "metric,value\n";
    file << "model," << model_name << "\n";
    file << "mode,image\n";
    file << "source," << images_dir << "\n";
    file << "samples," << samples << "\n";
    file << "total_time_s," << std::fixed << std::setprecision(4) << total_time_s << "\n";
    file << "fps_overall," << std::fixed << std::setprecision(4) << fps_overall << "\n";
    file << "time_ms_mean," << std::fixed << std::setprecision(4) << time_ms_mean << "\n";
    file << "time_ms_std," << std::fixed << std::setprecision(4) << time_ms_std << "\n";
    file << "fps_mean," << std::fixed << std::setprecision(4) << fps_mean << "\n";
    file << "fps_min," << std::fixed << std::setprecision(4) << fps_min << "\n";
    file << "fps_max," << std::fixed << std::setprecision(4) << fps_max << "\n";
    file << "detections_mean," << std::fixed << std::setprecision(4) << detections_mean << "\n";
    file.close();
}

static int run_images_inference(const Options& opts) {
    if (opts.images_dir.empty()) {
        std::cerr << "Error: --images-dir is required in images mode\n";
        return 1;
    }

    namespace fs = std::filesystem;
    fs::path images_path(opts.images_dir);
    if (!fs::exists(images_path) || !fs::is_directory(images_path)) {
        std::cerr << "Error: images-dir not found or not a directory: " << opts.images_dir << "\n";
        return 1;
    }

    // Collect image paths
    std::vector<fs::path> image_paths;
    for (const auto& ent : fs::directory_iterator(images_path)) {
        if (!ent.is_regular_file()) continue;
        auto ext = ent.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            image_paths.push_back(ent.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        std::cerr << "Error: no images found in " << opts.images_dir << "\n";
        return 1;
    }

    const size_t max_images = std::min<size_t>(image_paths.size(), (opts.frames > 0 ? (size_t)opts.frames : image_paths.size()));

    std::cout << "Running images inference...\n";
    std::cout << "Images: " << max_images << " (of " << image_paths.size() << " total)\n";

    neon::init_preprocess_buffers();

    InferenceEngine engine;
    InferenceEngine::Config engine_config;
    engine_config.param_path = opts.param_path;
    engine_config.bin_path = opts.bin_path;
    engine_config.use_int8 = opts.use_int8;
    engine_config.use_vulkan = opts.use_vulkan;
    engine_config.gpu_device = opts.gpu_device;
    engine_config.use_fp16 = !opts.use_int8;

    const char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        engine_config.num_threads = std::atoi(omp_threads);
    } else {
        engine_config.num_threads = NCNN_NUM_THREADS;
    }

    ErrorCode err = engine.initialize(engine_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }

    int warmup = opts.warmup_frames > 0 ? opts.warmup_frames : 1;
    engine.warmup(warmup);

    // Dynamically allocate model input buffer based on detected model width/height
    size_t model_input_floats = engine.get_model_width() * engine.get_model_height() * 3;
    AlignedPtr<float> model_input = make_aligned_buffer<float>(model_input_floats);
    if (!model_input) {
        std::cerr << "Failed to allocate input buffer\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }

    // For output
    std::vector<std::string> items;
    std::vector<double> time_ms;
    std::vector<double> fps;
    std::vector<int> detections;

    // Optional YOLO predictions export (for external mAP)
    if (!opts.preds_dir.empty()) {
        fs::create_directories(opts.preds_dir);
    }

    // Model name: use param stem
    std::string model_name = fs::path(opts.param_path).stem().string();

    for (size_t idx = 0; idx < max_images; idx++) {
        const fs::path& img_path = image_paths[idx];
        cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "Warning: cannot read image: " << img_path.string() << "\n";
            continue;
        }

        float scale = 1.0f;
        int pad_x = 0, pad_y = 0;

        int64_t total_start = get_timestamp_ns();

        // Preprocess (BGR path)
        neon::preprocess_bgr_direct(
            bgr.data,
            model_input.get(),
            bgr.cols,
            bgr.rows,
            engine.get_model_width(),
            engine.get_model_height(),
            (int)bgr.step[0],
            &scale, &pad_x, &pad_y
        );

        engine.set_letterbox_params(scale, pad_x, pad_y, bgr.cols, bgr.rows);

        DetectionResult result;
        ErrorCode infer_err = engine.infer_fp32(model_input.get(), result);
        (void)infer_err;
        int64_t total_end = get_timestamp_ns();

        // Filter detections by class if requested
        if (!opts.class_filter.empty()) {
            int write_idx = 0;
            for (int i = 0; i < result.count; i++) {
                if (opts.class_filter.count(result.detections[i].class_id) > 0) {
                    if (write_idx != i) {
                        result.detections[write_idx] = result.detections[i];
                    }
                    write_idx++;
                }
            }
            result.count = write_idx;
        }

        const int64_t total_time_us = (total_end - total_start) / 1000;
        const double ms = (double)total_time_us / 1000.0;
        const double cur_fps = (total_time_us > 0) ? (1000000.0 / (double)total_time_us) : 0.0;

        items.push_back(img_path.filename().string());
        time_ms.push_back(ms);
        fps.push_back(cur_fps);
        detections.push_back(result.count);

        if (!opts.preds_dir.empty()) {
            fs::path out_txt = fs::path(opts.preds_dir) / (img_path.stem().string() + ".txt");
            std::ofstream of(out_txt);
            if (of.is_open()) {
                for (int i = 0; i < result.count; i++) {
                    const auto& d = result.detections[i];
                    // Convert normalized x1y1x2y2 => YOLO x_center y_center w h (all normalized)
                    float xc = (d.x1 + d.x2) * 0.5f;
                    float yc = (d.y1 + d.y2) * 0.5f;
                    float w = (d.x2 - d.x1);
                    float h = (d.y2 - d.y1);
                    of << d.class_id << " " << xc << " " << yc << " " << w << " " << h << " " << d.confidence << "\n";
                }
            }
        }

        if ((idx + 1) % 50 == 0 || idx + 1 == max_images) {
            std::cout << "Processed " << (idx + 1) << "/" << max_images << "\n";
        }
    }

    // Stats
    const size_t samples = items.size();
    double total_time_s = 0.0;
    for (double v : time_ms) total_time_s += v / 1000.0;

    double time_ms_mean = 0.0;
    for (double v : time_ms) time_ms_mean += v;
    time_ms_mean /= (samples > 0 ? (double)samples : 1.0);

    double time_ms_var = 0.0;
    for (double v : time_ms) {
        double diff = v - time_ms_mean;
        time_ms_var += diff * diff;
    }
    time_ms_var /= (samples > 0 ? (double)samples : 1.0);
    double time_ms_std = std::sqrt(time_ms_var);

    double fps_mean = 0.0;
    for (double v : fps) fps_mean += v;
    fps_mean /= (samples > 0 ? (double)samples : 1.0);

    double fps_min = samples > 0 ? *std::min_element(fps.begin(), fps.end()) : 0.0;
    double fps_max = samples > 0 ? *std::max_element(fps.begin(), fps.end()) : 0.0;

    double detections_mean = 0.0;
    for (int c : detections) detections_mean += (double)c;
    detections_mean /= (samples > 0 ? (double)samples : 1.0);

    double fps_overall = (total_time_s > 0.0) ? ((double)samples / total_time_s) : 0.0;

    write_image_detail_csv(opts.output_image_detail_csv, items, time_ms, fps, detections);
    write_image_summary_csv(
        opts.output_image_summary_csv,
        model_name,
        opts.images_dir,
        samples,
        total_time_s,
        fps_overall,
        time_ms_mean,
        time_ms_std,
        fps_mean,
        fps_min,
        fps_max,
        detections_mean
    );

    neon::cleanup_preprocess_buffers();

    std::cout << "Done.\n";
    return 0;
}

// ============================================================================
// Test Functions
// ============================================================================

int test_model_loading(const Options& opts) {
    std::cout << "Testing model loading...\n";
    
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Model loaded successfully\n";
    return 0;
}

int test_single_inference(const Options& opts) {
    std::cout << "Testing single frame inference...\n";
    
    // Initialize engine
    InferenceEngine engine;
    InferenceEngine::Config config;
    config.param_path = opts.param_path;
    config.bin_path = opts.bin_path;
    config.num_threads = NCNN_NUM_THREADS;
    config.use_fp16 = true;
    
    ErrorCode err = engine.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        return 1;
    }
    
    // Warmup
    std::cout << "Warming up...\n";
    engine.warmup(3);
    
    // Create test input (all zeros)
    size_t model_input_size = engine.get_model_width() * engine.get_model_height() * 3;
    AlignedPtr<__fp16> input = make_aligned_buffer<__fp16>(model_input_size);
    memset(input.get(), 0, model_input_size * sizeof(__fp16));
    
    // Run inference
    DetectionResult result;
    int64_t inference_time;
    
    {
        ScopedTimer timer(inference_time);
        err = engine.infer(input.get(), result);
    }
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Inference failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Inference completed in " << inference_time << " us\n";
    std::cout << "Detections: " << result.count << "\n";
    
    return 0;
}

int test_camera_capture(const Options& opts) {
    std::cout << "Testing camera capture from " << opts.device << "...\n";
    
    InputPipeline pipeline;
    InputPipeline::Config config;
    config.source = InputSource::CAMERA_V4L2;
    config.device_path = opts.device;
    config.width = INPUT_WIDTH;
    config.height = INPUT_HEIGHT;
    config.fps = 30;
    
    ErrorCode err = pipeline.initialize(config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize camera: " << error_to_string(err) << "\n";
        return 1;
    }
    
    int frames_captured = 0;
    int target_frames = opts.frames > 0 ? opts.frames : 30;
    
    err = pipeline.start([&](const FrameBuffer& frame) -> bool {
        frames_captured++;
        std::cout << "Frame " << frames_captured << ": " 
                  << frame.width << "x" << frame.height 
                  << ", " << frame.size << " bytes\n";
        return frames_captured < target_frames && g_running.load();
    });
    
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Camera capture failed: " << error_to_string(err) << "\n";
        return 1;
    }
    
    std::cout << "Captured " << frames_captured << " frames\n";
    return 0;
}

// ============================================================================
// Main Processing Loop
// ============================================================================

int run_inference_pipeline(const Options& opts) {
    std::cout << "Starting OPTIMIZED inference pipeline...\n";
    std::cout << "Mode: " << opts.mode << "\n";
    std::cout << "Device/File: " << opts.device << "\n";
    if (!opts.output_video.empty()) {
        std::cout << "Output video: " << opts.output_video << "\n";
    }
    
    // Initialize preprocessing buffers ONCE
    neon::init_preprocess_buffers();
    
    // Initialize inference engine
    InferenceEngine engine;
    InferenceEngine::Config engine_config;
    engine_config.param_path = opts.param_path;
    engine_config.bin_path = opts.bin_path;
    
    // Get thread count from environment or use default
    const char* omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads) {
        engine_config.num_threads = std::atoi(omp_threads);
    } else {
        // OPTIMIZATION: Use fewer threads when display is active
        // This reduces CPU contention with X11/Qt display thread
        // Testing shows 3 threads + display = 26 FPS vs 4 threads + display = 22 FPS
        if (opts.display_enabled) {
            engine_config.num_threads = 3;  // Reserve 1 core for display
            std::cout << "Display mode: Using 3 NCNN threads (reduce contention)\n";
        } else {
            engine_config.num_threads = NCNN_NUM_THREADS;
        }
    }
    
    // INT8 and Vulkan options
    engine_config.use_int8 = opts.use_int8;
    engine_config.use_vulkan = opts.use_vulkan;
    engine_config.gpu_device = opts.gpu_device;
    
    // Don't use FP16 with INT8 mode
    engine_config.use_fp16 = !opts.use_int8;
    
    ErrorCode err = engine.initialize(engine_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to load model: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    // Print acceleration status
    std::cout << "Model loaded (threads=" << engine_config.num_threads;
    if (engine.is_using_int8()) std::cout << ", INT8";
    if (engine.is_using_vulkan()) std::cout << ", Vulkan";
    std::cout << "), warming up...\n";
    engine.warmup(30);  // Extended warmup for JIT/cache priming and stability
    
    // Initialize input pipeline
    InputPipeline pipeline;
    InputPipeline::Config input_config;
    
    if (opts.mode == "camera") {
        input_config.source = InputSource::CAMERA_V4L2;
        input_config.device_path = opts.device;
    } else {
        input_config.source = InputSource::VIDEO_FILE;
        input_config.device_path = opts.device;
        input_config.loop_video = opts.output_video.empty();  // Don't loop if saving video
    }
    
    input_config.width = INPUT_WIDTH;
    input_config.height = INPUT_HEIGHT;
    input_config.fps = 30;
    
    err = pipeline.initialize(input_config);
    if (err != ErrorCode::SUCCESS) {
        std::cerr << "Failed to initialize input: " << error_to_string(err) << "\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    // Initialize async video writer (if output video specified)
    std::unique_ptr<AsyncVideoWriter> video_writer;
    int video_width = INPUT_WIDTH;   // Use pipeline resolution (already resized)
    int video_height = INPUT_HEIGHT;
    
    if (!opts.output_video.empty()) {
        video_writer = std::make_unique<AsyncVideoWriter>();
        AsyncVideoWriter::Config writer_config;
        writer_config.output_path = opts.output_video;
        writer_config.width = video_width;
        writer_config.height = video_height;
        writer_config.fps = pipeline.get_video_fps();  // Match input video FPS
        // Queue size: buffer all frames (inference faster than encoding)
        // For typical videos: ~30fps * 60sec = 1800 frames max (~500MB)
        writer_config.queue_size = 2000;
        writer_config.draw_fps = opts.show_fps;
        
        if (!video_writer->start(writer_config)) {
            std::cerr << "Failed to initialize video writer\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
        
        std::cout << "Video writer initialized: " << video_width << "x" << video_height << "\n";
    }
    
    // Initialize async display (if enabled)
    std::unique_ptr<AsyncDisplay> display;
    if (opts.display_enabled) {
        display = std::make_unique<AsyncDisplay>();
        AsyncDisplay::Config display_config;
        display_config.window_name = "YOLOv8n Detection";
        display_config.queue_size = 3;  // Very small for low latency
        display_config.draw_fps = opts.show_fps;
        display_config.draw_bbox = true;
        display_config.max_screen_ratio = 0.75f;  // 75% of screen max
        
        if (!display->start(display_config, INPUT_WIDTH, INPUT_HEIGHT)) {
            std::cerr << "Failed to initialize display\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
    }
    
    // Initialize framebuffer display (if enabled) - bypasses X11, faster!
    std::unique_ptr<FramebufferDisplay> fb_display;
    if (opts.fb_display_enabled) {
        fb_display = std::make_unique<FramebufferDisplay>();
        FramebufferDisplay::Config fb_config;
        fb_config.target_width = INPUT_WIDTH;
        fb_config.target_height = INPUT_HEIGHT;
        fb_config.draw_fps = opts.show_fps;
        fb_config.draw_bbox = true;
        
        if (!fb_display->start(fb_config)) {
            std::cerr << "Failed to initialize framebuffer display\n";
            std::cerr << "Try: sudo chmod 666 /dev/fb0\n";
            neon::cleanup_preprocess_buffers();
            return 1;
        }
        std::cout << "Framebuffer mode: No X11 overhead, max FPS!\n";
    }
    
    // Initialize touch input (if framebuffer display is active)
    std::unique_ptr<TouchInput> touch_input;
    if (fb_display) {
        touch_input = std::make_unique<TouchInput>();
        
        // Set callback: forward touch/click events to fb_display for button hit-testing
        FramebufferDisplay* fb_ptr = fb_display.get();
        touch_input->set_touch_callback([fb_ptr](int x, int y, bool pressed) {
            fb_ptr->handle_input(x, y, pressed);
        });
        
        if (touch_input->start(fb_display->screen_width(), fb_display->screen_height())) {
            std::cout << "Touch input: active";
            if (touch_input->has_mouse()) std::cout << " (mouse cursor enabled)";
            std::cout << "\n";
        } else {
            std::cerr << "Warning: Touch input failed to start (check /dev/input permissions)\n";
            std::cerr << "  Try: sudo chmod 666 /dev/input/event*\n";
            touch_input.reset();
        }
    }
    
    // Initialize servo controller (if enabled + framebuffer active)
    std::unique_ptr<ServoController> servo;
    if (opts.servo_enabled && fb_display) {
        servo = std::make_unique<ServoController>();
        ServoController::Config servo_config;
        // Default: GPIO 12, PWM chip 0, channel 0
        servo_config.pwm_chip    = 0;
        servo_config.pwm_channel = 0;
        servo_config.speed_dps   = 90.0f;   // 90°/sec when holding button
        
        if (servo->start(servo_config)) {
            // Enable servo buttons on display
            fb_display->set_servo_enabled(true);
            fb_display->set_servo_callback([&servo](int dir) {
                servo->set_direction(dir);
            });
            std::cout << "Servo control: active (GPIO 12 PWM)\n";
        } else {
            std::cerr << "Warning: Servo init failed (see instructions above)\n";
            servo.reset();
        }
    }
    
    // Integrated MJPEG (same camera frames as inference; like CoralVisionRT)
    std::unique_ptr<MjpegStreamServer> mjpeg_srv;
    if (opts.mjpeg_port > 0) {
        mjpeg_srv = std::make_unique<MjpegStreamServer>(opts.mjpeg_port);
        if (!mjpeg_srv->start()) {
            std::cerr << "Warning: MJPEG server failed to bind (port in use?)\n";
            mjpeg_srv.reset();
        }
    }

    // Initialize Dashboard Client (if enabled)
    std::unique_ptr<DashboardClient> dashboard;
    if (!opts.dashboard_url.empty()) {
        dashboard = std::make_unique<DashboardClient>();
        if (dashboard->start(opts.dashboard_url)) {
            std::cout << "Dashboard Client: active -> " << opts.dashboard_url << "\n";

            std::string reg_url = opts.dashboard_stream_url;
            if (reg_url.empty() && mjpeg_srv) {
                std::string host = opts.tailscale_ip;
                if (host.empty()) {
                    host = detect_tailscale_ip_v4();
                }
                if (host.empty()) {
                    host = detect_local_ip_udp();
                }
                if (!host.empty()) {
                    reg_url = "http://" + host + ":" + std::to_string(opts.mjpeg_port) + "/stream";
                }
            }
            if (!reg_url.empty()) {
                if (dashboard->register_stream_url(reg_url)) {
                    std::cout << "Dashboard: stream URL registered -> " << reg_url << "\n";
                } else {
                    std::cout << "Warning: POST /stream-config failed (check FastAPI on " << opts.dashboard_url << ")\n";
                }
            }
        } else {
            std::cerr << "Warning: Dashboard Client failed to start\n";
            dashboard.reset();
        }
    }
    
    // Initialize benchmark
    Benchmark benchmark;
    BenchmarkConfig bench_config;
    bench_config.warmup_frames = opts.warmup_frames;
    bench_config.test_frames = opts.frames;
    bench_config.verbose = opts.verbose;
    benchmark.configure(bench_config);

    bool interval_logging_enabled = false;
    if (!opts.interval_output_csv.empty() && opts.snapshot_sec > 0) {
        interval_logging_enabled = initialize_interval_metrics_csv(opts.interval_output_csv);
        if (!interval_logging_enabled) {
            std::cerr << "Warning: interval metrics logging disabled\n";
        }
    }

    using SteadyClock = std::chrono::steady_clock;
    SteadyClock::time_point measurement_start_time;
    bool measurement_started = false;
    double last_snapshot_elapsed_sec = 0.0;
    double next_snapshot_elapsed_sec = static_cast<double>(opts.snapshot_sec);
    size_t last_snapshot_timing_count = 0;
    
    // Allocate FP32 model input buffer (pre-allocated, reused)
    size_t model_input_floats = engine.get_model_width() * engine.get_model_height() * 3;
    AlignedPtr<float> model_input = make_aligned_buffer<float>(model_input_floats);
    if (!model_input) {
        std::cerr << "Failed to allocate input buffer\n";
        neon::cleanup_preprocess_buffers();
        return 1;
    }
    
    std::cout << "Starting frame processing...\n";
    std::cout << "Warmup: " << opts.warmup_frames << " frames\n";
    std::cout << "Test: " << opts.frames << " frames\n\n";
    if (opts.duration_sec > 0) {
        std::cout << "Duration: " << opts.duration_sec << " sec after warmup\n";
    }
    if (interval_logging_enabled) {
        std::cout << "Interval metrics: every " << opts.snapshot_sec
                  << " sec -> " << opts.interval_output_csv << "\n";
    }
    if (opts.duration_sec > 0 || interval_logging_enabled) {
        std::cout << "\n";
    }
    
    // Track FPS for overlay
    float rolling_fps = 0;
    float rolling_inference_ms = 0;

    auto flush_interval_snapshot = [&](double elapsed_sec, bool force) {
        if (!measurement_started || !interval_logging_enabled) {
            return;
        }

        const size_t end_index = benchmark.timings().size();
        if (end_index <= last_snapshot_timing_count) {
            return;
        }

        if (!force && opts.snapshot_sec <= 0) {
            return;
        }

        BenchmarkStats window_stats = benchmark.calculate_stats_range(last_snapshot_timing_count, end_index);
        if (window_stats.total_frames <= 0) {
            return;
        }

        const double temperature_c = read_system_temperature_celsius();
        if (!append_interval_metrics_csv(
                opts.interval_output_csv,
                elapsed_sec,
                last_snapshot_elapsed_sec,
                elapsed_sec,
                window_stats,
                temperature_c)) {
            interval_logging_enabled = false;
            return;
        }

        std::cout << "\n[interval] " << std::fixed << std::setprecision(1) << elapsed_sec
                  << "s | fps_mean=" << std::setprecision(2) << window_stats.fps_mean
                  << " | p99_us=" << std::setprecision(1) << window_stats.p99_total_us
                  << " | temp=";
        if (std::isnan(temperature_c)) {
            std::cout << "n/a";
        } else {
            std::cout << std::setprecision(2) << temperature_c << "C";
        }
        std::cout << " -> " << opts.interval_output_csv << "\n";

        last_snapshot_timing_count = end_index;
        last_snapshot_elapsed_sec = elapsed_sec;
    };
    
    // Frame processing callback
    auto process_frame = [&](const FrameBuffer& frame) -> bool {
        FrameTiming timing;
        timing.frame_index = frame.frame_index;

        if (!measurement_started && benchmark.warmup_complete()) {
            measurement_started = true;
            measurement_start_time = SteadyClock::now();
            last_snapshot_elapsed_sec = 0.0;
            next_snapshot_elapsed_sec = static_cast<double>(opts.snapshot_sec);
            std::cout << "\nMeasurement window started after warmup\n";
        }
        
        int64_t total_start = get_timestamp_ns();
        
        timing.capture_time_us = 0;
        
        // Preprocessing - dispatch based on pixel format
        float scale;
        int pad_x, pad_y;
        int64_t preprocess_time = 0;
        
        {
            ScopedTimer timer(preprocess_time);
            
            if (frame.format == PixelFormat::BGR) {
                // OPTIMIZED: Direct BGR path (video files)
                neon::preprocess_bgr_direct(
                    frame.data,
                    model_input.get(),
                    frame.width,
                    frame.height,
                    engine.get_model_width(),
                    engine.get_model_height(),
                    frame.stride,
                    &scale, &pad_x, &pad_y
                );
            } else {
                // Camera path (YUYV)
                neon::preprocess_yuyv_to_fp32(
                    frame.data,
                    model_input.get(),
                    engine.get_model_width(),
                    engine.get_model_height(),
                    &scale, &pad_x, &pad_y
                );
            }
        }
        timing.preprocess_time_us = preprocess_time;
        
        // Set letterbox params for coordinate mapping
        engine.set_letterbox_params(scale, pad_x, pad_y, frame.width, frame.height);
        
        // Inference - DIRECT FP32
        DetectionResult result;
        int64_t inference_time = 0;
        
        {
            ScopedTimer timer(inference_time);
            
            ErrorCode infer_err = engine.infer_fp32(model_input.get(), result);
            if (infer_err != ErrorCode::SUCCESS) {
                std::cerr << "Inference failed on frame " << frame.frame_index << "\n";
            }
        }
        timing.inference_time_us = inference_time;
        
        timing.postprocess_time_us = 0;
        
        // Filter detections by class if specified
        if (!opts.class_filter.empty()) {
            int write_idx = 0;
            for (int i = 0; i < result.count; i++) {
                if (opts.class_filter.count(result.detections[i].class_id) > 0) {
                    if (write_idx != i) {
                        result.detections[write_idx] = result.detections[i];
                    }
                    write_idx++;
                }
            }
            result.count = write_idx;
        }
        
        // Total time
        int64_t total_end = get_timestamp_ns();
        timing.total_time_us = (total_end - total_start) / 1000;
        timing.detection_count = result.count;
        
        // Update rolling stats for FPS overlay
        float current_fps = 1000000.0f / timing.total_time_us;
        float current_inference_ms = inference_time / 1000.0f;
        rolling_fps = rolling_fps * 0.9f + current_fps * 0.1f;
        rolling_inference_ms = rolling_inference_ms * 0.9f + current_inference_ms * 0.1f;
        
        // Push to Dashboard (if active) — body must match FastAPI MetricIn
        // (see Dashboard-Detection-main/app/schemas.py: latency_ms, detect_count, etc.)
        if (dashboard) {
            try {
                const double temp_c = read_system_temperature_celsius();
                const double cpu_temp_for_api = std::isnan(temp_c) ? 0.0 : temp_c;
                nlohmann::json metrics = {
                    {"fps", rolling_fps},
                    {"latency_ms", rolling_inference_ms},
                    {"cpu_temp_c", cpu_temp_for_api},
                    {"cpu_percent", 0.0},
                    {"ram_percent", 0.0},
                    {"detect_count", result.count},
                    {"camera_status", "ok"},
                };
                dashboard->send_metrics(metrics.dump());

                auto draw_and_send_jpeg = [&](const cv::Mat& bgr) {
                    cv::Mat display_frame = bgr.clone();
                    for (int i = 0; i < result.count; i++) {
                        const auto& d = result.detections[i];
                        cv::rectangle(display_frame,
                            cv::Point(static_cast<int>(d.x1 * display_frame.cols),
                                      static_cast<int>(d.y1 * display_frame.rows)),
                            cv::Point(static_cast<int>(d.x2 * display_frame.cols),
                                      static_cast<int>(d.y2 * display_frame.rows)),
                            cv::Scalar(0, 255, 0), 2);
                    }
                    std::vector<uint8_t> jpeg_buffer;
                    const int q = std::max(30, std::min(95, opts.mjpeg_quality));
                    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, q};
                    cv::imencode(".jpg", display_frame, jpeg_buffer, encode_params);
                    dashboard->send_snapshot(jpeg_buffer);
                    if (mjpeg_srv) {
                        mjpeg_srv->update_frame(jpeg_buffer);
                    }
                };

                if (frame.format == PixelFormat::BGR) {
                    cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
                    draw_and_send_jpeg(bgr_frame);
                } else if (frame.format == PixelFormat::YUYV) {
                    cv::Mat yuyv(frame.height, frame.width, CV_8UC2, const_cast<uint8_t*>(frame.data), frame.stride);
                    cv::Mat bgr;
                    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
                    draw_and_send_jpeg(bgr);
                }
            } catch (...) {}
        }
        
        // Push frame to async writer (non-blocking, done AFTER inference)
        if (video_writer && frame.format == PixelFormat::BGR) {
            // Create cv::Mat wrapper (no copy, just wrap existing data)
            cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
            
            // Push to async queue (bbox drawing happens in writer thread)
            video_writer->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
        }
        
        // Push frame to async display (non-blocking)
        if (display) {
            if (frame.format == PixelFormat::BGR) {
                cv::Mat bgr_frame(frame.height, frame.width, CV_8UC3, frame.data, frame.stride);
                display->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
            } else if (frame.format == PixelFormat::YUYV) {
                // Convert YUYV to BGR for display
                cv::Mat yuyv_frame(frame.height, frame.width, CV_8UC2, frame.data, frame.stride);
                cv::Mat bgr_frame;
                cv::cvtColor(yuyv_frame, bgr_frame, cv::COLOR_YUV2BGR_YUYV);
                display->push(bgr_frame, result, frame.width, frame.height,
                             rolling_fps, rolling_inference_ms);
            }
        }
        
        // Push frame to framebuffer display (direct, no X11 overhead)
        if (fb_display) {
            // Get mouse cursor position (if available)
            int cursor_x = -1, cursor_y = -1;
            if (touch_input) {
                touch_input->get_cursor(cursor_x, cursor_y);
            }
            
            // Update servo angle display
            if (servo) {
                fb_display->update_servo_angle(servo->get_angle());
            }
            
            if (frame.format == PixelFormat::BGR) {
                fb_display->push_bgr(frame.data, frame.width, frame.height, frame.stride,
                                    result, rolling_fps, rolling_inference_ms,
                                    cursor_x, cursor_y);
            } else if (frame.format == PixelFormat::YUYV) {
                // Convert YUYV to BGR for framebuffer display
                cv::Mat yuyv_frame(frame.height, frame.width, CV_8UC2, frame.data, frame.stride);
                cv::Mat bgr_frame;
                cv::cvtColor(yuyv_frame, bgr_frame, cv::COLOR_YUV2BGR_YUYV);
                fb_display->push_bgr(bgr_frame.data, bgr_frame.cols, bgr_frame.rows, 
                                    bgr_frame.step, result, rolling_fps, rolling_inference_ms,
                                    cursor_x, cursor_y);
            }
        }

        // Dashboard web UI POSTs /servo; Pi must poll GET /servo (was never wired before).
        // Touch buttons on fb0 still win while held.
        if (dashboard && servo) {
            static uint64_t servo_poll_i = 0;
            if ((++servo_poll_i % 3u) == 0u) {
                int cmd = dashboard->fetch_servo_command();
                if (cmd >= -1 && cmd <= 1) {
                    const int touch_dir = fb_display ? fb_display->servo_touch_direction() : 0;
                    if (touch_dir == 0) {
                        servo->set_direction(cmd);
                    }
                }
            }
        }
        
        // Record timing
        benchmark.record_frame(timing);

        if (measurement_started) {
            const double elapsed_sec = std::chrono::duration<double>(
                SteadyClock::now() - measurement_start_time
            ).count();

            while (interval_logging_enabled &&
                   opts.snapshot_sec > 0 &&
                   elapsed_sec >= next_snapshot_elapsed_sec) {
                flush_interval_snapshot(elapsed_sec, false);
                next_snapshot_elapsed_sec += opts.snapshot_sec;
            }

            if (opts.duration_sec > 0 && elapsed_sec >= opts.duration_sec) {
                g_running.store(false);
            }
        }
        
        // Progress indicator
        if (!opts.verbose && benchmark.current_frame() % 100 == 0) {
            std::cout << "Frame " << benchmark.current_frame() 
                      << " | " << timing.total_time_us << "us"
                      << " | " << current_fps << " FPS";
            if (video_writer) {
                std::cout << " | Queue: " << video_writer->queue_size();
            }
            std::cout << "\r" << std::flush;
        }
        
        return !benchmark.is_complete() && g_running.load();
    };
    
    // Run pipeline
    err = pipeline.start(process_frame);

    if (measurement_started && interval_logging_enabled) {
        const double elapsed_sec = std::chrono::duration<double>(
            SteadyClock::now() - measurement_start_time
        ).count();
        flush_interval_snapshot(elapsed_sec, true);
    }
    
    std::cout << "\n";
    
    // Stop video writer (flushes remaining frames)
    if (video_writer) {
        std::cout << "Flushing video writer...\n";
        video_writer->stop();
        std::cout << "Video saved: " << opts.output_video << "\n";
        std::cout << "  Frames written: " << video_writer->frames_written() << "\n";
        std::cout << "  Frames dropped: " << video_writer->frames_dropped() << "\n";
    }
    
    // Stop servo controller
    if (servo) {
        servo->stop();
    }
    
    // Stop touch input
    if (touch_input) {
        touch_input->stop();
    }
    
    // Stop async display
    if (display) {
        display->stop();
    }
    
    // Stop framebuffer display
    if (fb_display) {
        fb_display->stop();
    }
    
    // Stop MJPEG before dashboard (closes HTTP clients)
    mjpeg_srv.reset();

    // Stop dashboard client
    if (dashboard) {
        dashboard->stop();
    }
    
    // Print results
    benchmark.print_summary();
    
    // Export CSV if requested
    if (!opts.output_csv.empty()) {
        benchmark.export_csv(opts.output_csv);
        std::cout << "Results exported to " << opts.output_csv << "\n";
    }
    
    // Cleanup
    neon::cleanup_preprocess_buffers();
    
    // Determine exit code based on validation
    BenchmarkStats stats = benchmark.calculate_stats();
    
    if (stats.is_valid()) {
        std::cout << "\n✓ SYSTEM MEETS ALL PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " >= 20\n";
        return 0;
    } else {
        std::cout << "\n✗ SYSTEM DOES NOT MEET PERFORMANCE REQUIREMENTS\n";
        std::cout << "  FPS (P99): " << stats.fps_p99 << " < 20\n";
        return 1;
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse options
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }
    
    // Run appropriate test/mode
    if (opts.test_model) {
        return test_model_loading(opts);
    }
    
    if (opts.test_inference) {
        return test_single_inference(opts);
    }
    
    if (opts.test_camera) {
        return test_camera_capture(opts);
    }

    if (opts.mode == "images") {
        return run_images_inference(opts);
    }
    
    // Run full pipeline
    return run_inference_pipeline(opts);
}
