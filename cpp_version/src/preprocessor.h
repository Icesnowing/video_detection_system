#pragma once

#include "common.h"

/**
 * Image preprocessing: BGR -> RGB -> resize (letterbox) -> normalize -> CHW tensor.
 *
 * Produces a float32 tensor ready for ONNX Runtime inference.
 * Uses zero-copy cv::Mat wrapping where possible.
 */
class Preprocessor {
public:
    Preprocessor(int input_width, int input_height);
    ~Preprocessor() = default;

    struct PreprocessResult {
        std::vector<float> tensor;      // Flattened (1, 3, H, W) float32 data
        std::vector<int64_t> shape;     // {1, 3, input_height, input_width}
        float ratio_w;
        float ratio_h;
        int pad_left;
        int pad_top;
    };

    PreprocessResult process(const cv::Mat& frame);

private:
    void letterbox(const cv::Mat& src, cv::Mat& dst,
                   float& ratio_w, float& ratio_h,
                   int& pad_left, int& pad_top);

    int m_input_width;
    int m_input_height;
};
