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
    result.shape = {1, 3, m_input_height, m_input_width}; //4维张量，NCHW 格式（PyTorch、Caffe 等框架使用）
    size_t total = 3 * m_input_height * m_input_width;
    result.tensor.resize(total);

    /*
    原始OpenCV HWC布局（Height-Width-Channel）：
    内存布局（按行存储）：
    行0: [B00,G00,R00, B01,G01,R01, B02,G02,R02, ...]
    行1: [B10,G10,R10, B11,G11,R11, B12,G12,R12, ...]
    行2: [B20,G20,R20, B21,G21,R21, B22,G22,R22, ...]
    
    CHW布局（Channel-Height-Width）：
    内存布局（按通道存储）：
    通道B: [B00, B01, B02, ..., B10, B11, B12, ..., B20, B21, B22, ...]
    通道G: [G00, G01, G02, ..., G10, G11, G12, ..., G20, G21, G22, ...]
    通道R: [R00, R01, R02, ..., R10, R11, R12, ..., R20, R21, R22, ...]
    
    */
    // Unroll HWC to CHW
    for (int c = 0; c < 3; c++) { //所有像素按通道连续存储
        for (int h = 0; h < m_input_height; h++) {
            for (int w = 0; w < m_input_width; w++) {
                size_t idx = c * m_input_height * m_input_width + h * m_input_width + w;
                //cv::Vec3f：包含 3 个 float 值的向量 [R, G, B]
                //.at<cv::Vec3f>(h, w)：访问第 h 行 w 列的像素，返回一个 cv::Vec3f 向量
                result.tensor[idx] = rgb.at<cv::Vec3f>(h, w)[c]; //获取第 c 个通道的像素值
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
