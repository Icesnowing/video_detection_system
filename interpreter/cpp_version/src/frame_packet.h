#pragma once

#include "common.h"
#include "preprocessor.h"

#include <cstdint>
#include <vector>

struct FramePacket {
    bool eos = false;

    uint32_t frame_idx = 0;
    int64_t pts = 0;
    double pts_sec = 0.0;
    int64_t duration = 0;
    bool is_keyframe = false;

    cv::Mat frame;
    bool run_detection = false;

    Preprocessor::PreprocessResult preproc;
    std::vector<float> output_data;
    std::vector<int64_t> output_shape;
    std::vector<Detection> detections;

    double decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double encode_ms = 0.0;

    static FramePacket make_eos() {
        FramePacket p;
        p.eos = true;
        return p;
    }
};
