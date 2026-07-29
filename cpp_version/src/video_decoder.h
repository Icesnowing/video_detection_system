#pragma once

#include "common.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/display.h>
}

#include <string>
#include <functional>
#include <deque>

/**
 * FFmpeg-based video decoder with:
 *   - Hardware decoding (VAAPI / CUDA)
 *   - Precise PTS timestamps
 *   - Frame-rate jitter handling
 *   - Frame-drop / corruption tolerance
 *   - Optionally outputs raw YUV to save conversion overhead
 */
class VideoDecoder {
public:
    VideoDecoder(const Config& cfg);
    ~VideoDecoder();

    bool open();
    bool read(DecodedFrame& frame);
    void close();

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }
    size_t frame_count() const { return m_display_idx; }
    AVRational time_base() const { return m_time_base; }

private:
    enum HwAccel { HW_NONE, HW_VAAPI, HW_CUDA };

    bool open_input();
    bool find_video_stream();
    bool setup_decoder();
    bool init_hw_decoder();
    bool setup_sws();

    bool decode_packet();
    bool receive_and_convert(DecodedFrame& out);
    bool convert_to_bgr(const AVFrame* src, cv::Mat& dst);
    double pts_to_sec(int64_t pts) const;
    bool fix_jitter(DecodedFrame& frame);

    Config m_cfg;
    HwAccel m_hw_type = HW_NONE;

    int m_width = 0;
    int m_height = 0;
    int m_codec_width = 0;
    int m_codec_height = 0;
    AVPixelFormat m_src_pix_fmt = AV_PIX_FMT_NONE;
    double m_fps = 0.0;
    int m_rotation = 0;

    AVFormatContext* m_fmt_ctx = nullptr;
    AVCodecContext* m_codec_ctx = nullptr;
    const AVCodec* m_codec = nullptr;
    int m_video_stream_idx = -1;

    AVBufferRef* m_hw_ctx = nullptr;
    AVBufferRef* m_hw_frames_ctx = nullptr;
    SwsContext* m_sws_ctx = nullptr;

    AVFrame* m_frame = nullptr;
    AVFrame* m_hw_frame = nullptr;
    AVFrame* m_bgr_frame = nullptr;

    AVPacket* m_pkt = nullptr;
    AVRational m_time_base{1, 1};
    bool m_is_open = false;

    // Timing / jitter state
    int64_t m_last_pts = AV_NOPTS_VALUE;
    int64_t m_expected_next_pts = AV_NOPTS_VALUE;
    uint32_t m_decode_idx = 0;
    uint32_t m_display_idx = 0;
    int m_consecutive_drops = 0;
    int m_consecutive_errors = 0;
    bool m_eof = false;

    // Jitter buffer
    std::deque<DecodedFrame> m_frame_buf;
    static constexpr size_t kMaxBufSize = 4;
};
