#pragma once

#include "common.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <vector>

class YuvEncoder {
public:
    YuvEncoder(const Config& cfg);
    ~YuvEncoder();

    bool open(int width, int height);
    bool write(const cv::Mat& frame);
    void close();

    size_t frame_count() const { return m_frame_count; }

private:
    size_t calc_frame_size() const;

    Config m_cfg;
    int m_width = 0;
    int m_height = 0;
    size_t m_frame_size = 0;
    size_t m_frame_count = 0;
    int m_cvt_code = -1;

    FILE* m_file = nullptr;
    std::vector<uint8_t> m_yuv_buf;
    cv::Mat m_yuv_mat;
    bool m_is_open = false;
};
