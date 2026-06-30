#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

struct Config {
    // Model
    std::string onnx_path = "models/yolo11n.onnx";
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = 300;
    int input_width = 640;
    int input_height = 640;

    // Source
    std::string source_type = "video";  // "usb_camera", "video", "rtsp"
    std::string source_path = "0";
    int capture_width = 1280;
    int capture_height = 720;
    int capture_fps = 30;

    // Decoder
    std::string hw_accel = "none";
    std::string hw_device = "";

    // Inference
    std::string provider = "CPUExecutionProvider";
    int intra_op_threads = 0;
    int inter_op_threads = 0;
    bool enable_profiling = false;

    // Encoder
    std::string output_path = "video_output/output.mp4";
    std::string codec = "libx264";
    std::string preset = "fast";
    int crf = 23;
    int output_fps = 30;
    std::string rtsp_push_url = "";

    // Display
    bool show_window = true;
    std::string window_name = "Video Detection (C++)";
    bool draw_labels = true;
    int box_thickness = 2;
    float font_scale = 0.5f;
};
