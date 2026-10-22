#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
    int64_t pts = 0;        // PTS timestamp aligned with source frame
    double pts_sec = 0.0;   // PTS in seconds
    int frame_idx = 0;
};

struct DecodedFrame {
    cv::Mat data;           // BGR image data (decoded from YUV)
    int64_t pts = 0;        // Presentation timestamp (in stream time_base units)
    double pts_sec = 0.0;   // PTS converted to seconds
    int64_t duration = 0;   // Frame duration (in time_base units)
    uint32_t frame_idx = 0; // decoded frame index
    bool is_keyframe = false;
    bool valid = false;
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
    std::string source_type = "video";  // "usb_camera", "video", "rtsp", "yuv"
    std::string source_path = "0";
    int capture_width = 1280;
    int capture_height = 720;
    int capture_fps = 30;

    // YUV raw input
    std::string yuv_format = "i420";    // "i420", "nv12", "nv21", "yuyv"
    int yuv_width = 1280;
    int yuv_height = 720;

    // YUV raw output (if output path has .yuv extension, use raw YUV)
    std::string yuv_output_format = "i420";

    // Decoder
    std::string hw_accel = "none";      // "vaapi", "cuda", "qsv", "none"
    std::string hw_device = "";         // e.g. "/dev/dri/renderD128" for VAAPI
    bool decoder_output_yuv = false;    // internal YUV output (avoids BGR conv overhead)
    double max_jitter_ms = 50.0;        // max PTS jitter tolerated before frame correction
    int max_consecutive_drops = 3;      // max consecutive dropped/corrupt frames before abort

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

    // Frame sampling (skip detection on some frames)
    std::string sample_mode = "none";     // "none", "keyframe", "skip", "interval"
    int sample_skip_n = 10;               // "skip": run detection every N frames
    double sample_interval_sec = 1.0;     // "interval": run detection every N seconds

    // Empty: video_detect_<local create time>.log. --log sets an explicit path.
    std::string log_path;

    // Display
    bool show_window = true;
    std::string window_name = "Video Detection (C++)";
    bool draw_labels = true;
    int box_thickness = 2;
    float font_scale = 0.5f;
};
