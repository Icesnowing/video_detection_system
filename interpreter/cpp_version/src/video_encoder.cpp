#include "video_encoder.h"
#include <iostream>
#include <cmath>

VideoEncoder::VideoEncoder(const Config& cfg) : m_cfg(cfg) {
    m_pkt = av_packet_alloc();
    m_yuv_frame = av_frame_alloc();
}

VideoEncoder::~VideoEncoder() { close(); }

bool VideoEncoder::open(int width, int height) {
    m_enc_width = width;
    m_enc_height = height;

    if (!init_format()) return false;
    if (!init_codec()) return false;

    m_is_open = true;
    std::cout << "[Encoder] Opened: " << m_enc_width << "x" << m_enc_height
              << " -> " << m_cfg.output_path << std::endl;
    return true;
}

bool VideoEncoder::init_format() {
    std::string out_path = m_cfg.output_path;
    if (!m_cfg.rtsp_push_url.empty())
        out_path = m_cfg.rtsp_push_url;

    int ret = avformat_alloc_output_context2(
        &m_fmt_ctx, nullptr, nullptr, out_path.c_str());
    if (ret < 0 || !m_fmt_ctx) {
        std::cerr << "[Encoder] Failed to create output context" << std::endl;
        return false;
    }
    return true;
}

bool VideoEncoder::init_codec() {
    m_codec = avcodec_find_encoder_by_name(m_cfg.codec.c_str());
    if (!m_codec)
        m_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!m_codec) {
        std::cerr << "[Encoder] Codec " << m_cfg.codec << " not found" << std::endl;
        return false;
    }

    m_stream = avformat_new_stream(m_fmt_ctx, nullptr);
    m_codec_ctx = avcodec_alloc_context3(m_codec);

    m_codec_ctx->width  = m_enc_width;
    m_codec_ctx->height = m_enc_height;
    m_codec_ctx->time_base = av_make_q(1, 90000);  // 90kHz clock
    m_codec_ctx->framerate  = av_make_q(m_cfg.output_fps, 1);
    m_codec_ctx->pix_fmt    = AV_PIX_FMT_YUV420P;
    m_codec_ctx->gop_size   = m_cfg.output_fps;
    m_codec_ctx->max_b_frames = 2;

    if (m_cfg.crf >= 0)
        av_opt_set_int(m_codec_ctx->priv_data, "crf", m_cfg.crf, 0);
    if (!m_cfg.preset.empty())
        av_opt_set(m_codec_ctx->priv_data, "preset", m_cfg.preset.c_str(), 0);

    if (m_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        m_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    int ret = avcodec_open2(m_codec_ctx, m_codec, nullptr);
    if (ret < 0) {
        char eb[256]; av_strerror(ret, eb, sizeof(eb));
        std::cerr << "[Encoder] avcodec_open2: " << eb << std::endl;
        return false;
    }

    avcodec_parameters_from_context(m_stream->codecpar, m_codec_ctx);

    if (!(m_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&m_fmt_ctx->pb, m_cfg.output_path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            std::cerr << "[Encoder] avio_open failed" << std::endl;
            return false;
        }
    }

    ret = avformat_write_header(m_fmt_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "[Encoder] avformat_write_header failed" << std::endl;
        return false;
    }

    m_yuv_frame->format = AV_PIX_FMT_YUV420P;
    m_yuv_frame->width  = m_enc_width;
    m_yuv_frame->height = m_enc_height;
    av_frame_get_buffer(m_yuv_frame, 0);

    m_sws_ctx = sws_getContext(
        m_enc_width, m_enc_height, AV_PIX_FMT_BGR24,
        m_enc_width, m_enc_height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    m_enc_pts = 0;
    return true;
}

int64_t VideoEncoder::to_enc_pts(double pts_sec, uint32_t frame_idx) {
    if (pts_sec >= 0.0) {
        return (int64_t)(pts_sec * 90000.0 + 0.5);
    }
    int64_t frame_duration = 90000 / m_cfg.output_fps;
    if (frame_duration < 1) frame_duration = 3000;
    return frame_idx * frame_duration;
}

bool VideoEncoder::write(const cv::Mat& frame, double pts_sec, uint32_t frame_idx) {
    if (!m_is_open) return false;

    const uint8_t* src[1] = { frame.data };
    int linesize[1] = { (int)frame.step };
    sws_scale(m_sws_ctx, src, linesize, 0, m_enc_height,
              m_yuv_frame->data, m_yuv_frame->linesize);

    m_yuv_frame->pts = to_enc_pts(pts_sec, frame_idx);

    int ret = avcodec_send_frame(m_codec_ctx, m_yuv_frame);
    if (ret < 0) return false;

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_codec_ctx, m_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) return false;

        av_packet_rescale_ts(m_pkt, m_codec_ctx->time_base, m_stream->time_base);
        m_pkt->stream_index = m_stream->index;
        av_interleaved_write_frame(m_fmt_ctx, m_pkt);
        av_packet_unref(m_pkt);
    }

    m_frame_count++;
    m_last_frame_idx = frame_idx;
    return true;
}

void VideoEncoder::close() {
    if (!m_is_open) return;
    m_is_open = false;

    avcodec_send_frame(m_codec_ctx, nullptr);
    int ret;
    do {
        ret = avcodec_receive_packet(m_codec_ctx, m_pkt);
        if (ret == 0) {
            av_packet_rescale_ts(m_pkt, m_codec_ctx->time_base, m_stream->time_base);
            m_pkt->stream_index = m_stream->index;
            av_interleaved_write_frame(m_fmt_ctx, m_pkt);
            av_packet_unref(m_pkt);
        }
    } while (ret == 0);

    av_write_trailer(m_fmt_ctx);

    if (m_sws_ctx) { sws_freeContext(m_sws_ctx); m_sws_ctx = nullptr; }
    if (m_codec_ctx) { avcodec_free_context(&m_codec_ctx); }
    if (m_fmt_ctx) {
        if (!(m_fmt_ctx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&m_fmt_ctx->pb);
        avformat_free_context(m_fmt_ctx);
        m_fmt_ctx = nullptr;
    }
    if (m_pkt) { av_packet_free(&m_pkt); }
    if (m_yuv_frame) { av_frame_free(&m_yuv_frame); }

    std::cout << "[Encoder] Closed. Total frames: " << m_frame_count << std::endl;
}
