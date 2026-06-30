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
 * FFmpeg-based video encoder.
 *
 * Encodes BGR frames to H.264/H.265 and writes to file,
 * with optional RTSP push streaming.
 */
class VideoEncoder {
public:
    VideoEncoder(const Config& cfg);
    ~VideoEncoder();

    bool open();
    bool write(const cv::Mat& frame);
    void close();

    size_t frame_count() const { return m_frame_count; }

private:
    bool init_codec();
    bool init_format();

    Config m_cfg;
    size_t m_frame_count = 0;

    AVFormatContext* m_fmt_ctx = nullptr;
    AVCodecContext* m_codec_ctx = nullptr;
    const AVCodec* m_codec = nullptr;
    AVStream* m_stream = nullptr;

    SwsContext* m_sws_ctx = nullptr;
    AVFrame* m_yuv_frame = nullptr;

    AVPacket* m_pkt = nullptr;
    bool m_is_open = false;
    int64_t m_pts = 0;
};
