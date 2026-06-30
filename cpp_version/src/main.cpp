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
 *   ./video_detection --benchmark
 */

#include "pipeline.h"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --source TYPE       Video source: usb_camera, video, rtsp (default: video)\n"
              << "  --path PATH         Camera index, file path, or RTSP URL\n"
              << "  --model PATH        ONNX model path (default: models/yolo11n.onnx)\n"
              << "  --confidence FLOAT  Confidence threshold (default: 0.25)\n"
              << "  --nms FLOAT         NMS IoU threshold (default: 0.45)\n"
              << "  --width INT         Capture width (default: 1280)\n"
              << "  --height INT        Capture height (default: 720)\n"
              << "  --no-display        Disable preview window\n"
              << "  --output PATH       Output video path\n"
              << "  --benchmark         Run inference benchmark\n"
              << "  --help              Show this help\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    Config cfg;

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
        } else if (arg == "--benchmark") {
            // Benchmark mode
            InferenceEngine engine(cfg);
            if (engine.load()) {
                engine.warmup(10);
                engine.benchmark(100);
            }
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (cfg.source_path.empty()) {
        std::cerr << "Error: --path is required." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    DetectionPipeline pipeline(cfg);

    try {
        pipeline.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
