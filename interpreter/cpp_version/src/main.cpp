/**
 * Video Detection System - C++ Version (CPU, ONNX Runtime)
 *
 * Features:
 *   - FFmpeg hardware/software video decoding (libav*)
 *   - OpenCV image preprocessing with zero-copy Mat wrapping
 *   - ONNX Runtime CPU inference (replaces TensorRT when no GPU)
 *   - Hand-written NMS post-processing and box rendering
 *   - FFmpeg H.264/H.265 encoding with optional RTSP push
 *
 * Usage:
 *   ./video_detection --source video --path test.mp4
 *   ./video_detection --source usb_camera --path /dev/video0
 *   ./video_detection --source rtsp --path rtsp://192.168.1.100:554/stream
 *   ./video_detection --source yuv --path input.yuv --yuv-format i420 --yuv-size 1920x1080
 *   ./video_detection --benchmark
 */

#include "pipeline.h"
#include "trace_log.h"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --source TYPE       Video source: usb_camera, video, rtsp, yuv (default: video)\n"
              << "  --path PATH         Camera index, file path, RTSP URL, or YUV file path\n"
              << "  --model PATH        ONNX model path (default: models/yolo11n.onnx)\n"
              << "  --confidence FLOAT  Confidence threshold (default: 0.25)\n"
              << "  --nms FLOAT         NMS IoU threshold (default: 0.45)\n"
              << "  --class NAME        Label drawn on boxes (default: car)\n"
              << "  --width INT         Capture width (default: 1280)\n"
              << "  --height INT        Capture height (default: 720)\n"
              << "  --yuv-format FMT    YUV format: i420, nv12, nv21, yuyv (default: i420)\n"
              << "  --yuv-size WxH      YUV frame dimensions, e.g. 1920x1080\n"
              << "  --no-display        Disable preview window\n"
              << "  --output PATH       Output video path (use .yuv extension for raw YUV)\n"
              << "  --codec NAME        Output codec: libx264, libx265 (default: libx264)\n"
              << "  --sample MODE       Frame sampling: none, keyframe, skip, interval\n"
              << "  --sample-skip N     Skip: run detection every N frames (default: 10)\n"
              << "  --sample-interval S Interval: run detection every S seconds (default: 1.0)\n"
              << "  --fps INT           Output frame rate for encoder (default: 30)\n"
              << "  --log PATH          Log file (default: video_detect_<create time>.log)\n"
              << "  --benchmark         Run inference benchmark\n"
              << "  --help              Show this help\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    Config cfg;
    bool run_benchmark = false;
    bool show_help = false;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--source" && i + 1 < argc) {
            cfg.source_type = argv[++i];
        } else if (arg == "--path" && i + 1 < argc) {
            cfg.source_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            cfg.onnx_path = argv[++i];
        } else if (arg == "--confidence" && i + 1 < argc) {
            cfg.confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            cfg.nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--class" && i + 1 < argc) {
            cfg.class_name = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            cfg.capture_width = std::stoi(argv[++i]);
            cfg.input_width = cfg.capture_width;
        } else if (arg == "--height" && i + 1 < argc) {
            cfg.capture_height = std::stoi(argv[++i]);
            cfg.input_height = cfg.capture_height;
        } else if (arg == "--no-display") {
            cfg.show_window = false;
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--codec" && i + 1 < argc) {
            cfg.codec = argv[++i];
        } else if (arg == "--yuv-format" && i + 1 < argc) {
            cfg.yuv_format = argv[++i];
        } else if (arg == "--yuv-size" && i + 1 < argc) {
            std::string size_str = argv[++i];
            size_t x = size_str.find('x');
            if (x != std::string::npos) {
                cfg.yuv_width = std::stoi(size_str.substr(0, x));
                cfg.yuv_height = std::stoi(size_str.substr(x + 1));
            }
        } else if (arg == "--fps" && i + 1 < argc) {
            cfg.output_fps = std::stoi(argv[++i]);
        } else if (arg == "--sample" && i + 1 < argc) {
            cfg.sample_mode = argv[++i];
        } else if (arg == "--sample-skip" && i + 1 < argc) {
            cfg.sample_skip_n = std::stoi(argv[++i]);
        } else if (arg == "--sample-interval" && i + 1 < argc) {
            cfg.sample_interval_sec = std::stod(argv[++i]);
        } else if (arg == "--hw-accel" && i + 1 < argc) {
            cfg.hw_accel = argv[++i];
        } else if (arg == "--hw-device" && i + 1 <argc) {
            cfg.hw_device = argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        }
    }

    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (!trace_log_open(cfg.log_path.empty() ? nullptr : cfg.log_path.c_str())) {
        LOGE("Error: cannot open log file: %s", g_trace_log_path.c_str());
        return 1;
    }
    cfg.log_path = g_trace_log_path;
    LOGI("Writing log to %s", cfg.log_path.c_str());
    //把config的配置参数写入日志
    LOGI("source_type: %s, source_path: %s, onnx_path: %s, confidence_threshold: %.4f, nms_threshold: %.4f, class_name: %s, capture_width: %d, capture_height: %d, "
        "output_path: %s, codec: %s, yuv_format: %s, yuv_size: %d*%d, "
        "output_fps: %d, sample_mode: %s, sample_skip_n: %d, sample_interval_sec: %.2f, hw_accel: %s, hw_device: %s", 
        cfg.source_type.c_str(), cfg.source_path.c_str(), cfg.onnx_path.c_str(), cfg.confidence_threshold, cfg.nms_threshold, cfg.class_name.c_str(),
        cfg.capture_width, cfg.capture_height, cfg.output_path.c_str(), cfg.codec.c_str(), cfg.yuv_format.c_str(), 
        cfg.yuv_width, cfg.yuv_height, cfg.output_fps, cfg.sample_mode.c_str(), cfg.sample_skip_n, cfg.sample_interval_sec, 
        cfg.hw_accel.c_str(), cfg.hw_device.c_str());
    if (run_benchmark) {
        {
            InferenceEngine engine(cfg);
            if (engine.load()) {
                engine.warmup(10);
                engine.benchmark(100);
            }
        }
        trace_log_close();
        return 0;
    }

    if (cfg.source_path.empty()) {
        LOGE("--path is required.");
        print_usage(argv[0]);
        trace_log_close();
        return 1;
    }

    int rc = 0;
    {
        DetectionPipeline pipeline(cfg);
        try {
            pipeline.run();
        } catch (const std::exception& e) {
            LOGE("Fatal error: %s", e.what());
            rc = 1;
        }
    }

    trace_log_close();
    return rc;
}
