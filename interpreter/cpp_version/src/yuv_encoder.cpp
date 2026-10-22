#include "yuv_encoder.h"
#include "trace_log.h"
#include <cstring>

YuvEncoder::YuvEncoder(const Config& cfg) : m_cfg(cfg) {}

YuvEncoder::~YuvEncoder() { close(); }

size_t YuvEncoder::calc_frame_size() const {
    if (m_cfg.yuv_output_format == "yuyv" || m_cfg.yuv_output_format == "yuy2") {
        return (size_t)m_width * m_height * 2;
    } else {
        return (size_t)m_width * m_height * 3 / 2;
    }
}

static void i420_to_nv12(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                          uint8_t* dst, int w, int h) {
    int y_size = w * h;
    std::memcpy(dst, y, y_size);
    uint8_t* uv = dst + y_size;
    int uv_size = w * h / 4;
    for (int i = 0; i < uv_size; i++) {
        uv[i * 2] = u[i];
        uv[i * 2 + 1] = v[i];
    }
}

static void i420_to_nv21(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                          uint8_t* dst, int w, int h) {
    int y_size = w * h;
    std::memcpy(dst, y, y_size);
    uint8_t* vu = dst + y_size;
    int uv_size = w * h / 4;
    for (int i = 0; i < uv_size; i++) {
        vu[i * 2] = v[i];
        vu[i * 2 + 1] = u[i];
    }
}

static void i420_to_yuyv(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                          uint8_t* dst, int w, int h) {
    int uv_w = w / 2;
    for (int row = 0; row < h; row++) {
        int y_row_off = row * w;
        int uv_row = row / 2;
        int uv_row_off = uv_row * uv_w;
        for (int col = 0; col < uv_w; col++) {
            int dst_off = row * w * 2 + col * 4;
            dst[dst_off + 0] = y[y_row_off + col * 2];
            dst[dst_off + 1] = u[uv_row_off + col];
            dst[dst_off + 2] = y[y_row_off + col * 2 + 1];
            dst[dst_off + 3] = v[uv_row_off + col];
        }
    }
}

bool YuvEncoder::open(int width, int height) {
    m_width = width;
    m_height = height;
    m_frame_size = calc_frame_size();

    if (m_width <= 0 || m_height <= 0) {
        LOGE("Invalid dimensions");
        return false;
    }

    m_file = fopen(m_cfg.output_path.c_str(), "wb");
    if (!m_file) {
        LOGE("Cannot open: %s", m_cfg.output_path.c_str());
        return false;
    }

    std::string fmt = m_cfg.yuv_output_format;
    if (fmt == "i420" || fmt == "iyuv" || fmt == "yuv420p") {
        m_cvt_code = cv::COLOR_BGR2YUV_I420;
    } else if (fmt == "nv12" || fmt == "nv21" || fmt == "yuyv" || fmt == "yuy2") {
        m_cvt_code = cv::COLOR_BGR2YUV_I420;  // convert to I420 first, then rearrange
    } else {
        LOGE("Unsupported format: %s", fmt.c_str());
        fclose(m_file);
        m_file = nullptr;
        return false;
    }

    int i420_h = m_height * 3 / 2;
    m_yuv_mat = cv::Mat(i420_h, m_width, CV_8UC1);
    m_yuv_buf.resize(m_frame_size);
    m_is_open = true;

    LOGI("Opened: %s %dx%d format=%s", m_cfg.output_path.c_str(), m_width, m_height, fmt.c_str());
    return true;
}

bool YuvEncoder::write(const cv::Mat& frame) {
    if (!m_is_open) return false;

    cv::cvtColor(frame, m_yuv_mat, cv::COLOR_BGR2YUV_I420);

    std::string fmt = m_cfg.yuv_output_format;
    if (fmt == "nv12") {
        const uint8_t* y = m_yuv_mat.ptr(0);
        const uint8_t* u = m_yuv_mat.ptr(m_height);
        const uint8_t* v = m_yuv_mat.ptr(m_height + m_height / 4);
        i420_to_nv12(y, u, v, m_yuv_buf.data(), m_width, m_height);
    } else if (fmt == "nv21") {
        const uint8_t* y = m_yuv_mat.ptr(0);
        const uint8_t* u = m_yuv_mat.ptr(m_height);
        const uint8_t* v = m_yuv_mat.ptr(m_height + m_height / 4);
        i420_to_nv21(y, u, v, m_yuv_buf.data(), m_width, m_height);
    } else if (fmt == "yuyv" || fmt == "yuy2") {
        const uint8_t* y = m_yuv_mat.ptr(0);
        const uint8_t* u = m_yuv_mat.ptr(m_height);
        const uint8_t* v = m_yuv_mat.ptr(m_height + m_height / 4);
        i420_to_yuyv(y, u, v, m_yuv_buf.data(), m_width, m_height);
    } else {
        std::memcpy(m_yuv_buf.data(), m_yuv_mat.data, m_frame_size);
    }

    fwrite(m_yuv_buf.data(), 1, m_frame_size, m_file);
    m_frame_count++;
    return true;
}

void YuvEncoder::close() {
    if (!m_is_open) return;
    m_is_open = false;
    if (m_file) {
        fclose(m_file);
        m_file = nullptr;
    }
    LOGI("Closed. Total frames: %zu", m_frame_count);
}
