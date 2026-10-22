#include "yuv_decoder.h"
#include "trace_log.h"
#include <cstring>

YuvDecoder::YuvDecoder(const Config& cfg) : m_cfg(cfg) {}

YuvDecoder::~YuvDecoder() { close(); }

size_t YuvDecoder::calc_frame_size() const {
    if (m_cfg.yuv_format == "yuyv" || m_cfg.yuv_format == "yuy2") {
        return (size_t)m_width * m_height * 2;
    } else {
        return (size_t)m_width * m_height * 3 / 2;
    }
}

bool YuvDecoder::open() {
    m_width = m_cfg.yuv_width;
    m_height = m_cfg.yuv_height;
    m_frame_size = calc_frame_size();

    if (m_width <= 0 || m_height <= 0) {
        LOGE("Invalid dimensions: %dx%d", m_width, m_height);
        return false;
    }

    m_file = fopen(m_cfg.source_path.c_str(), "rb");
    if (!m_file) {
        LOGE("Cannot open: %s", m_cfg.source_path.c_str());
        return false;
    }

    std::string fmt = m_cfg.yuv_format;
    if (fmt == "i420" || fmt == "iyuv" || fmt == "yuv420p") {
        m_cvt_code = cv::COLOR_YUV2BGR_I420;
        m_yuv_mat = cv::Mat(m_height * 3 / 2, m_width, CV_8UC1);
    } else if (fmt == "nv12") {
        m_cvt_code = cv::COLOR_YUV2BGR_NV12;
        m_yuv_mat = cv::Mat(m_height * 3 / 2, m_width, CV_8UC1);
    } else if (fmt == "nv21") {
        m_cvt_code = cv::COLOR_YUV2BGR_NV21;
        m_yuv_mat = cv::Mat(m_height * 3 / 2, m_width, CV_8UC1);
    } else if (fmt == "yuyv" || fmt == "yuy2") {
        m_cvt_code = cv::COLOR_YUV2BGR_YUYV;
        m_yuv_mat = cv::Mat(m_height, m_width, CV_8UC2);
    } else {
        LOGE("Unsupported format: %s", fmt.c_str());
        fclose(m_file);
        m_file = nullptr;
        return false;
    }

    m_yuv_buf.resize(m_frame_size);
    m_is_open = true;

    LOGI("Opened: %s %dx%d format=%s framesize=%zu",
         m_cfg.source_path.c_str(), m_width, m_height, fmt.c_str(), m_frame_size);
    return true;
}

bool YuvDecoder::read(cv::Mat& frame) {
    if (!m_is_open) return false;

    size_t n = fread(m_yuv_buf.data(), 1, m_frame_size, m_file);
    if (n != m_frame_size) {
        if (n == 0) return false;
        LOGE("Short read: %zu/%zu", n, m_frame_size);
        return false;
    }

    std::memcpy(m_yuv_mat.data, m_yuv_buf.data(), m_frame_size);
    cv::cvtColor(m_yuv_mat, frame, m_cvt_code);
    m_frame_count++;
    return true;
}

void YuvDecoder::close() {
    m_is_open = false;
    if (m_file) {
        fclose(m_file);
        m_file = nullptr;
    }
    LOGI("Closed. Total frames: %zu", m_frame_count);
}
