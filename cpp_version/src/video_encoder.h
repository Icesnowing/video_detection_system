#pragma once

#include "common.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

/**
 * FFmpeg-based video encoder with PTS-aware timing.
 *
 * Encodes BGR frames to H.264/H.265 with proper PTS alignment,
 * writes to file, with optional RTSP push streaming.
 */
class VideoEncoder {
public:
    VideoEncoder(const Config& cfg);
    ~VideoEncoder();

    bool open(int width, int height);
    bool write(const cv::Mat& frame, double pts_sec = -1.0, uint32_t frame_idx = 0);
    void close();

    size_t frame_count() const { return m_frame_count; }

private:
    bool init_codec();
    bool init_format();
    int64_t to_enc_pts(double pts_sec, uint32_t frame_idx);

    Config m_cfg;
    int m_enc_width = 0;
    int m_enc_height = 0;
    size_t m_frame_count = 0;

    AVFormatContext* m_fmt_ctx = nullptr;
    AVCodecContext* m_codec_ctx = nullptr;
    const AVCodec* m_codec = nullptr;
    AVStream* m_stream = nullptr;

    SwsContext* m_sws_ctx = nullptr;
    AVFrame* m_yuv_frame = nullptr;

    AVPacket* m_pkt = nullptr;
    bool m_is_open = false;

    // PTS tracking
    int64_t m_enc_pts = 0;
    uint32_t m_last_frame_idx = 0;
};
