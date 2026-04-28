/**
 * @file inference_engine.cpp
 * @brief NCNN-based YOLOv8n inference engine - RADICALLY OPTIMIZED
 * 
 * CRITICAL OPTIMIZATIONS:
 * - Direct FP32 input (no FP16 conversion overhead)
 * - Optimized thread count (2 threads for better cache sharing)
 * - Vectorized output transpose
 * - Zero-copy where possible
 * - INT8 quantization support for 2-4x speedup
 * - Vulkan GPU support for VideoCore VII (RPi5)
 */

#include "inference_engine.h"
#include "postprocess.h"
#include "asm_kernels.h"

#include <net.h>
#include <cpu.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#include <cmath>
#include <cstring>
#include <fstream>
#include <arm_neon.h>
#include <iostream>
#include <sstream>

// #region agent log
#warning "DEBUG_SYNC_a1fd6f_inference_engine_cpp"
// #endregion

namespace yolo {

namespace {

int64_t debug_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

std::string debug_escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char ch : input) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

void debug_log(const char* run_id,
               const char* hypothesis_id,
               const char* location,
               const char* message,
               const std::string& data_json) {
    std::ofstream log_file("debug-a1fd6f.log", std::ios::app);
    if (!log_file.is_open()) {
        return;
    }

    log_file
        << "{\"sessionId\":\"a1fd6f\""
        << ",\"runId\":\"" << run_id << "\""
        << ",\"hypothesisId\":\"" << hypothesis_id << "\""
        << ",\"location\":\"" << location << "\""
        << ",\"message\":\"" << message << "\""
        << ",\"data\":" << data_json
        << ",\"timestamp\":" << debug_timestamp_ms()
        << "}\n";
}

}  // namespace

// ============================================================================
// InferenceEngine Implementation
// ============================================================================

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() {
    if (net_) {
        delete net_;
    }
    if (blob_pool_allocator_) {
        delete blob_pool_allocator_;
    }
    if (workspace_allocator_) {
        delete workspace_allocator_;
    }
#if NCNN_VULKAN
    if (blob_vkallocator_) {
        delete blob_vkallocator_;
    }
    if (staging_vkallocator_) {
        delete staging_vkallocator_;
    }
#endif
}

ErrorCode InferenceEngine::initialize(const Config& config) {
    config_ = config;

    // #region agent log
    debug_log(
        "repro-1",
        "H4",
        "src/inference_engine.cpp:initialize:entry",
        "initialize called",
        std::string("{\"use_int8\":") + (config.use_int8 ? "true" : "false") +
        ",\"use_fp16\":" + (config.use_fp16 ? std::string("true") : std::string("false")) +
        ",\"num_threads\":" + std::to_string(config.num_threads) +
        ",\"param_path\":\"" + debug_escape_json(config.param_path) + "\"" +
        ",\"bin_path\":\"" + debug_escape_json(config.bin_path) + "\"}"
    );
    // #endregion
    
    // Configure NCNN global settings
    ncnn::set_cpu_powersave(0);  // Use all cores at full speed
    ncnn::set_omp_num_threads(config.num_threads);
    
#if NCNN_VULKAN
    // Initialize Vulkan if requested
    if (config.use_vulkan) {
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0 && config.gpu_device < gpu_count) {
            using_vulkan_ = true;
            std::cout << "Vulkan GPU: " << ncnn::get_gpu_info(config.gpu_device).device_name() 
                      << " (device " << config.gpu_device << "/" << gpu_count << ")\n";
        } else {
            std::cout << "Vulkan requested but no GPU found, falling back to CPU\n";
        }
    }
#endif

    // Create NCNN pool allocators
    blob_pool_allocator_ = new ncnn::PoolAllocator();
    workspace_allocator_ = new ncnn::UnlockedPoolAllocator();
    
    // Pre-allocate pools to avoid runtime allocation
    blob_pool_allocator_->set_size_compare_ratio(0.0f);  // Exact match only
    workspace_allocator_->set_size_compare_ratio(0.0f);
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(config.gpu_device);
        blob_vkallocator_ = new ncnn::VkBlobAllocator(vkdev);
        staging_vkallocator_ = new ncnn::VkStagingAllocator(vkdev);
    }
#endif

    // Create and configure network
    net_ = new ncnn::Net();
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        net_->opt.use_vulkan_compute = true;
        net_->set_vulkan_device(config.gpu_device);
    }
#endif
    
    net_->opt.lightmode = config.light_mode;
    net_->opt.num_threads = config.num_threads;
    net_->opt.use_packing_layout = config.use_packing;
    
    // INT8 specific settings
    if (config.use_int8) {
        using_int8_ = true;
        net_->opt.use_int8_inference = true;
        net_->opt.use_int8_storage = true;
        net_->opt.use_int8_arithmetic = true;
        // Disable FP16 when using INT8
        net_->opt.use_fp16_packed = false;
        net_->opt.use_fp16_storage = false;
        net_->opt.use_fp16_arithmetic = false;
        std::cout << "INT8 quantization enabled\n";
    } else {
        net_->opt.use_fp16_packed = config.use_fp16;
        net_->opt.use_fp16_storage = config.use_fp16;
        net_->opt.use_fp16_arithmetic = config.use_fp16;
    }
    
    net_->opt.blob_allocator = blob_pool_allocator_;
    net_->opt.workspace_allocator = workspace_allocator_;
    
#if NCNN_VULKAN
    if (using_vulkan_) {
        net_->opt.blob_vkallocator = blob_vkallocator_;
        net_->opt.workspace_vkallocator = blob_vkallocator_;
        net_->opt.staging_vkallocator = staging_vkallocator_;
    }
#endif
    
    // Load model
    int ret = net_->load_param(config.param_path.c_str());
    if (ret != 0) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    ret = net_->load_model(config.bin_path.c_str());
    if (ret != 0) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    // Get input/output blob names - auto-detect from model
    const auto& input_names = net_->input_names();
    const auto& output_names = net_->output_names();
    
    if (!input_names.empty()) {
        input_name_ = input_names[0];
    } else {
        input_name_ = "images";
    }
    
    if (!output_names.empty()) {
        output_name_ = output_names[0];
    } else {
        output_name_ = "output0";
    }

    // Determine model width and height dynamically by examining the input layer
    const auto& blobs = net_->blobs();
    bool used_fallback_model_size = false;
    for (size_t i = 0; i < blobs.size(); i++) {
        if (blobs[i].name == input_name_) {
            model_width_ = blobs[i].shape.w;
            model_height_ = blobs[i].shape.h;
            break;
        }
    }
    // Fallback if not specified in model
    if (model_width_ <= 0 || model_height_ <= 0) {
        model_width_ = 640; 
        model_height_ = 640;
        used_fallback_model_size = true;
    }
    
    // #region agent log
    debug_log(
        "repro-1",
        "H1",
        "src/inference_engine.cpp:initialize:model-meta",
        "model io metadata resolved",
        std::string("{\"input_name\":\"") + debug_escape_json(input_name_) +
        "\",\"output_name\":\"" + debug_escape_json(output_name_) +
        "\",\"model_width\":" + std::to_string(model_width_) +
        ",\"model_height\":" + std::to_string(model_height_) +
        ",\"used_fallback_size\":" +
        (used_fallback_model_size ? std::string("true") : std::string("false")) +
        "}"
    );
    // #endregion

    std::cout << "[DEBUG] Detected model input: " << model_width_ << "x" << model_height_ << std::endl;
    
    // Run a quick dummy inference to figure out output tensor dimensions (num_classes)
    {
        ncnn::Mat dummy_input(model_width_, model_height_, 3);
        dummy_input.fill(0.0f);
        ncnn::Extractor ex = net_->create_extractor();
        ex.set_light_mode(config_.light_mode);
        ex.input(input_name_.c_str(), dummy_input);
        ncnn::Mat dummy_output;
        int dummy_ret = ex.extract(output_name_.c_str(), dummy_output);
        
        std::cout << "[DEBUG] Dummy inference ret: " << dummy_ret << std::endl;
        std::cout << "[DEBUG] Dummy output shape: c=" << dummy_output.c << " h=" << dummy_output.h << " w=" << dummy_output.w << std::endl;
        
        if (dummy_ret != 0 || dummy_output.w == 0 || dummy_output.h == 0) {
            std::cerr << "Failed to extract dummy output. Falling back to default dimensions." << std::endl;
            num_classes_ = 1; // Fallback
            int num_outputs = 8400; // Fallback
            output_buffer_size_ = num_outputs * (4 + num_classes_);
        } else {
            // Check if shape is 1x84x8400 (c=1, h=84, w=8400) or c=84, h=8400, w=1
            int h = dummy_output.h;
            int w = dummy_output.w;
            int c = dummy_output.c;
            
            // Normally YOLOv8 NCNN export is h=84 (channels), w=8400 (proposals)
            // If c=84 and w=8400 and h=1, we need to handle that.
            int proposals = w;
            int channels = h;
            
            if (c > 1 && h == 1 && w > 1) { // e.g. c=84, h=1, w=8400
                channels = c;
                proposals = w;
            } else if (c == 1 && h > 1 && w > 1) { // e.g. c=1, h=84, w=8400
                channels = h;
                proposals = w;
            } else if (c > 1 && h > 1 && w == 1) { // e.g. c=8400, h=84, w=1
                channels = h;
                proposals = c;
            }

            int inferred_input_size = 0;
            const double stage_squared = static_cast<double>(proposals) / 21.0;
            const int stage = static_cast<int>(std::lround(std::sqrt(stage_squared)));
            if (stage > 0 && 21 * stage * stage == proposals) {
                inferred_input_size = stage * 32;
            }

            const bool inconsistent_with_output =
                inferred_input_size > 0 &&
                (model_width_ != inferred_input_size || model_height_ != inferred_input_size);

            if (inferred_input_size > 0 && (used_fallback_model_size || inconsistent_with_output)) {
                model_width_ = inferred_input_size;
                model_height_ = inferred_input_size;
                std::cout << "[DEBUG] Corrected model input from output proposals: "
                          << model_width_ << "x" << model_height_ << std::endl;
            }

            num_classes_ = channels - 4; // YOLOv8 has 4 bbox outputs + C class outputs
            output_buffer_size_ = proposals * channels;
            
            std::cout << "[DEBUG] Detected proposals: " << proposals << ", channels: " << channels << ", classes: " << num_classes_ << std::endl;
        }

        // #region agent log
        debug_log(
            "repro-1",
            "H2",
            "src/inference_engine.cpp:initialize:dummy-output",
            "dummy inference output analyzed",
            std::string("{\"dummy_ret\":") + std::to_string(dummy_ret) +
            ",\"dims\":" + std::to_string(dummy_output.dims) +
            ",\"c\":" + std::to_string(dummy_output.c) +
            ",\"h\":" + std::to_string(dummy_output.h) +
            ",\"w\":" + std::to_string(dummy_output.w) +
            ",\"elempack\":" + std::to_string(dummy_output.elempack) +
            ",\"num_classes\":" + std::to_string(num_classes_) +
            ",\"final_model_width\":" + std::to_string(model_width_) +
            ",\"final_model_height\":" + std::to_string(model_height_) +
            ",\"output_buffer_size\":" + std::to_string(output_buffer_size_) +
            "}"
        );
        // #endregion
    }
    
    output_buffer_ = make_aligned_buffer<float>(output_buffer_size_);
    
    if (!output_buffer_) {
        return ErrorCode::MEMORY_ALLOCATION_FAILED;
    }
    
    initialized_ = true;
    return ErrorCode::SUCCESS;
}

// ============================================================================
// OPTIMIZED: Direct FP32 inference (no conversion!)
// ============================================================================

ErrorCode InferenceEngine::infer_fp32(const float* input_data, DetectionResult& result) {
    if (!initialized_) {
        return ErrorCode::MODEL_LOAD_FAILED;
    }
    
    result.count = 0;
    
    // Create NCNN Mat directly from FP32 data (ZERO-COPY!)
    // NCNN Mat uses CHW layout, same as our input
    ncnn::Mat input(model_width_, model_height_, 3, (void*)input_data);
    
    // Run inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(config_.light_mode);
    
    ex.input(input_name_.c_str(), input);
    
    ncnn::Mat output;
    int ret = ex.extract(output_name_.c_str(), output);
    
    if (ret != 0) {
        return ErrorCode::INFERENCE_FAILED;
    }
    
    // Parse dimensions correctly based on output shape (c, h, w)
    int proposals = output.w;
    int channels = output.h;
    
    if (output.c > 1 && output.h == 1 && output.w > 1) { 
        channels = output.c;
        proposals = output.w;
    } else if (output.c == 1 && output.h > 1 && output.w > 1) { 
        channels = output.h;
        proposals = output.w;
    } else if (output.c > 1 && output.h > 1 && output.w == 1) {
        channels = output.h;
        proposals = output.c;
    }

    // #region agent log
    debug_log(
        "repro-1",
        "H3",
        "src/inference_engine.cpp:infer_fp32:before-transpose",
        "output extracted before transpose",
        std::string("{\"dims\":") + std::to_string(output.dims) +
        ",\"c\":" + std::to_string(output.c) +
        ",\"h\":" + std::to_string(output.h) +
        ",\"w\":" + std::to_string(output.w) +
        ",\"elempack\":" + std::to_string(output.elempack) +
        ",\"proposals\":" + std::to_string(proposals) +
        ",\"channels\":" + std::to_string(channels) +
        ",\"buffer_size\":" + std::to_string(output_buffer_size_) +
        ",\"output_ptr\":" + std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(output.data))) +
        "}"
    );
    // #endregion
    
    // OPTIMIZED: Assembly transpose [channels, proposals] -> [proposals, channels]
    // Use hand-tuned assembly for transpose
    transpose_84x8400_asm(
        (const float*)output.data,
        output_buffer_.get(),
        proposals,
        channels
    );

    // #region agent log
    debug_log(
        "repro-1",
        "H3",
        "src/inference_engine.cpp:infer_fp32:after-transpose",
        "transpose completed",
        std::string("{\"proposals\":") + std::to_string(proposals) +
        ",\"channels\":" + std::to_string(channels) +
        ",\"buffer_size\":" + std::to_string(output_buffer_size_) +
        ",\"dst_ptr\":" + std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(output_buffer_.get()))) +
        "}"
    );
    // #endregion
    
    // Decode outputs
    result.count = decode_yolov8_output(
        output_buffer_.get(),
        proposals,
        num_classes_,
        model_width_,
        CONF_THRESHOLD,
        result.detections,
        MAX_DETECTIONS
    );
    
    // Apply NMS
    if (result.count > 0) {
        sort_detections_by_confidence(result.detections, result.count);
        nms_sorted(result.detections, result.count, NMS_THRESHOLD);
    }
    
    // Map coordinates back to original image space
    for (int i = 0; i < result.count; i++) {
        map_detection_to_original(
            result.detections[i],
            scale_, pad_x_, pad_y_,
            orig_width_, orig_height_,
            model_width_
        );
    }
    
    return ErrorCode::SUCCESS;
}

// Legacy FP16 interface (now just converts and calls FP32)
ErrorCode InferenceEngine::infer(const __fp16* input_data, DetectionResult& result) {
    // Convert FP16 to FP32 for NCNN
    size_t model_input_floats = model_width_ * model_height_ * 3;
    AlignedPtr<float> fp32_input = make_aligned_buffer<float>(model_input_floats);
    
    const size_t channel_size = model_width_ * model_height_;
    
    for (int c = 0; c < 3; c++) {
        float* dst = fp32_input.get() + c * channel_size;
        const __fp16* src = input_data + c * channel_size;
        
        size_t i = 0;
        for (; i + 8 <= channel_size; i += 8) {
            float16x8_t fp16_vec = vld1q_f16(src + i);
            float32x4_t fp32_lo = vcvt_f32_f16(vget_low_f16(fp16_vec));
            float32x4_t fp32_hi = vcvt_f32_f16(vget_high_f16(fp16_vec));
            vst1q_f32(dst + i, fp32_lo);
            vst1q_f32(dst + i + 4, fp32_hi);
        }
        for (; i < channel_size; i++) {
            dst[i] = static_cast<float>(src[i]);
        }
    }
    
    return infer_fp32(fp32_input.get(), result);
}

void InferenceEngine::warmup(int iterations) {
    if (!initialized_) return;
    
    // Create dummy input
    size_t model_input_floats = model_width_ * model_height_ * 3;
    AlignedPtr<float> dummy_input = make_aligned_buffer<float>(model_input_floats);
    memset(dummy_input.get(), 0, model_input_floats * sizeof(float));
    
    DetectionResult dummy_result;

    // #region agent log
    debug_log(
        "repro-1",
        "H1",
        "src/inference_engine.cpp:warmup:start",
        "warmup starting",
        std::string("{\"iterations\":") + std::to_string(iterations) +
        ",\"model_width\":" + std::to_string(model_width_) +
        ",\"model_height\":" + std::to_string(model_height_) +
        ",\"model_input_floats\":" + std::to_string(model_input_floats) +
        ",\"dummy_input_ptr\":" +
        std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(dummy_input.get()))) +
        "}"
    );
    // #endregion
    
    for (int i = 0; i < iterations; i++) {
        infer_fp32(dummy_input.get(), dummy_result);
    }
}

}  // namespace yolo
