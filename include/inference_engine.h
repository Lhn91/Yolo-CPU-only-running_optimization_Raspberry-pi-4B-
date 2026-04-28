/**
 * @file inference_engine.h
 * @brief NCNN-based YOLOv8n inference engine
 *
 * Inference & NCNN Agent Design:
 * - Static NCNN build for Cortex-A76 with FP16/INT8
 * - Thread pool pinned to CPU 2-3 (avoiding input thread on CPU 0-1)
 * - Pre-allocated blob memory to avoid runtime allocation
 * - Custom allocator for deterministic memory behavior
 * - Vulkan GPU support for VideoCore VII (RPi5)
 * - INT8 quantization support for 2-4x speedup
 * - Fully dynamic resolution and class count (No more fixed MODEL_SIZE)
 */

#ifndef YOLO_INFERENCE_ENGINE_H
#define YOLO_INFERENCE_ENGINE_H

#include "common.h"
#include <string>
#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace yolo {

/**
 * @brief YOLOv8n inference engine using NCNN
 */
class InferenceEngine {
public:
    /**
     * @brief Engine configuration
     */
    struct Config {
        std::string param_path;     // Path to .param file
        std::string bin_path;       // Path to .bin file
        int num_threads = NCNN_NUM_THREADS;
        bool use_fp16 = true;       // Use FP16 inference
        bool use_packing = true;    // Use NEON packing
        int light_mode = 1;         // NCNN light mode
        bool use_vulkan = false;    // Use Vulkan GPU compute
        bool use_int8 = false;      // Use INT8 quantized model
        int gpu_device = 0;         // Vulkan GPU device index
    };

    InferenceEngine();
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    ErrorCode initialize(const Config& config);
    ErrorCode infer(const __fp16* input_data, DetectionResult& result);
    ErrorCode infer_fp32(const float* input_data, DetectionResult& result);

    void set_letterbox_params(
        float scale,
        int pad_x,
        int pad_y,
        int orig_width = INPUT_WIDTH,
        int orig_height = INPUT_HEIGHT
    ) {
        scale_ = scale;
        pad_x_ = pad_x;
        pad_y_ = pad_y;
        orig_width_ = orig_width;
        orig_height_ = orig_height;
    }

    bool is_initialized() const { return initialized_; }
    bool is_using_vulkan() const { return using_vulkan_; }
    bool is_using_int8() const { return using_int8_; }

    int get_model_width() const { return model_width_; }
    int get_model_height() const { return model_height_; }
    int get_num_classes() const { return num_classes_; }

    void warmup(int iterations = 3);

private:
    void decode_outputs(const float* output_data, int output_size, DetectionResult& result);
    void apply_nms(DetectionResult& result);

    bool initialized_ = false;
    bool using_vulkan_ = false;
    bool using_int8_ = false;
    Config config_;

    ncnn::Net* net_ = nullptr;
    ncnn::PoolAllocator* blob_pool_allocator_ = nullptr;
    ncnn::UnlockedPoolAllocator* workspace_allocator_ = nullptr;
#if NCNN_VULKAN
    ncnn::VkAllocator* blob_vkallocator_ = nullptr;
    ncnn::VkAllocator* staging_vkallocator_ = nullptr;
#endif

    AlignedPtr<float> output_buffer_;
    size_t output_buffer_size_ = 0;

    std::string input_name_;
    std::string output_name_;

    int model_width_ = 0;
    int model_height_ = 0;
    int num_classes_ = 0;

    float scale_ = 1.0f;
    int pad_x_ = 0;
    int pad_y_ = 0;
    int orig_width_ = INPUT_WIDTH;
    int orig_height_ = INPUT_HEIGHT;
};

}  // namespace yolo

#endif  // YOLO_INFERENCE_ENGINE_H
