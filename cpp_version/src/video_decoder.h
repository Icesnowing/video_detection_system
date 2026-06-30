#pragma once

#include "common.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
#include <libavutil/display.h>
}

#include <string>
#include <functional>

/**
 * FFmpeg-based hardware/software video decoder.
 *
 * Opens any video source (file, camera, RTSP) via FFmpeg,
 * decodes to raw BGR frames, and provides a pull-based interface.
 */
class VideoDecoder {
public:
    VideoDecoder(const Config& cfg);
    ~VideoDecoder();

    bool open();
    bool read(cv::Mat& frame);
    void close();

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }
    size_t frame_count() const { return m_frame_count; }

private:
    bool open_input();
    bool find_video_stream();
    bool setup_decoder();
    bool convert_to_bgr(const AVFrame* src, cv::Mat& dst);

    Config m_cfg;
    int m_width = 0;
    int m_height = 0;
    int m_codec_width = 0;
    int m_codec_height = 0;
    double m_fps = 0.0;
    size_t m_frame_count = 0;
    int m_rotation = 0;

    AVFormatContext* m_fmt_ctx = nullptr;
    AVCodecContext* m_codec_ctx = nullptr;
    const AVCodec* m_codec = nullptr;
    int m_video_stream_idx = -1;

    SwsContext* m_sws_ctx = nullptr;
    AVFrame* m_frame = nullptr;
    AVFrame* m_bgr_frame = nullptr;

    AVPacket* m_pkt = nullptr;
    bool m_is_open = false;
};
