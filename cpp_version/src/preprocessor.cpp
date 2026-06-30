#include "preprocessor.h"
#include <algorithm>

Preprocessor::Preprocessor(int input_width, int input_height)
    : m_input_width(input_width), m_input_height(input_height) {}

Preprocessor::PreprocessResult Preprocessor::process(const cv::Mat& frame) {
    PreprocessResult result;

    cv::Mat letterboxed;
    letterbox(frame, letterboxed,
              result.ratio_w, result.ratio_h,
              result.pad_left, result.pad_top);

    // BGR -> RGB, convert to float, normalize to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW layout
    result.shape = {1, 3, m_input_height, m_input_width};
    size_t total = 3 * m_input_height * m_input_width;
    result.tensor.resize(total);

    // Unroll HWC to CHW
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < m_input_height; h++) {
            for (int w = 0; w < m_input_width; w++) {
                size_t idx = c * m_input_height * m_input_width + h * m_input_width + w;
                result.tensor[idx] = rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return result;
}

void Preprocessor::letterbox(const cv::Mat& src, cv::Mat& dst,
                              float& ratio_w, float& ratio_h,
                              int& pad_left, int& pad_top) {
    int h0 = src.rows;
    int w0 = src.cols;

    float r_h = (float)m_input_height / h0;
    float r_w = (float)m_input_width / w0;
    float ratio = std::min(r_h, r_w);

    ratio_w = ratio;
    ratio_h = ratio;

    int new_w = (int)std::round(w0 * ratio);
    int new_h = (int)std::round(h0 * ratio);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int dw = m_input_width - new_w;
    int dh = m_input_height - new_h;

    pad_left = dw / 2;
    pad_top = dh / 2;
    int pad_right = dw - pad_left;
    int pad_bottom = dh - pad_top;

    cv::copyMakeBorder(resized, dst,
                       pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
}
