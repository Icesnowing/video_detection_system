#include "video_decoder.h"
#include "trace_log.h"
#include <cstring>
#include <cmath>

VideoDecoder::VideoDecoder(const Config& cfg) : m_cfg(cfg) {
    if (m_cfg.hw_accel == "vaapi") m_hw_type = HW_VAAPI;
    else if (m_cfg.hw_accel == "cuda") m_hw_type = HW_CUDA;

    m_pkt = av_packet_alloc();
    m_frame = av_frame_alloc();
    m_bgr_frame = av_frame_alloc();
    if (m_hw_type != HW_NONE) m_hw_frame = av_frame_alloc();
}

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open() {
    if (!open_input()) return false;
    if (!find_video_stream()) return false;
    if (!setup_decoder()) return false;
    if (!setup_sws()) return false;

    m_eof = false;
    m_is_open = true;
    LOGI("Opened: %dx%d @ %.2ffps hw=%s", m_width, m_height, m_fps, m_cfg.hw_accel.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Open input
// ---------------------------------------------------------------------------
bool VideoDecoder::open_input() {
    AVDictionary* opts = nullptr;
    std::string src = m_cfg.source_path;
    std::string fmt_name;

    if (m_cfg.source_type == "usb_camera") {
        fmt_name = "v4l2";
        av_dict_set(&opts, "framerate", std::to_string(m_cfg.capture_fps).c_str(), 0);
        av_dict_set(&opts, "video_size",
                    (std::to_string(m_cfg.capture_width) + "x" +
                     std::to_string(m_cfg.capture_height)).c_str(), 0);
    } else if (m_cfg.source_type == "rtsp") {
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "stimeout", "5000000", 0);  // 5s timeout
    }

    const AVInputFormat* input_fmt = nullptr;
    if (!fmt_name.empty())
        input_fmt = av_find_input_format(fmt_name.c_str());

    int ret = avformat_open_input(&m_fmt_ctx, src.c_str(), input_fmt, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char eb[256]; av_strerror(ret, eb, sizeof(eb));
        LOGE("avformat_open_input: %s", eb);
        return false;
    }

    ret = avformat_find_stream_info(m_fmt_ctx, nullptr);
    if (ret < 0) {
        LOGE("avformat_find_stream_info failed");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Find video stream + extract metadata
// ---------------------------------------------------------------------------
bool VideoDecoder::find_video_stream() {
    m_video_stream_idx = -1;
    for (unsigned i = 0; i < m_fmt_ctx->nb_streams; i++) {
        if (m_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { // find video stream
            m_video_stream_idx = i;
            break;
        }
    }
    if (m_video_stream_idx < 0) {
        LOGE("No video stream");
        return false;
    }

    AVStream* stream = m_fmt_ctx->streams[m_video_stream_idx];
    AVCodecParameters* params = stream->codecpar;

    m_codec_width  = params->width;
    m_codec_height = params->height;
    m_src_pix_fmt  = (AVPixelFormat)params->format;
    m_time_base    = stream->time_base;

    if (m_codec_width <= 0 || m_codec_height <= 0) {
        m_codec_width  = m_cfg.capture_width;
        m_codec_height = m_cfg.capture_height;
    }

    // Rotation metadata
    for (int i = 0; i < params->nb_coded_side_data; i++) {
        const auto& sd = params->coded_side_data[i];
        if (sd.type == AV_PKT_DATA_DISPLAYMATRIX && sd.size >= 9 * 4) {
            double r = av_display_rotation_get((const int32_t*)sd.data);
            int deg = (int)(r + 360.0) % 360;
            if (deg == 90 || deg == 270) m_rotation = deg;
            else if (deg == 180) m_rotation = 180;
        }
    }

    if (m_rotation == 90 || m_rotation == 270) {
        m_width  = m_codec_height;
        m_height = m_codec_width;
    } else {
        m_width  = m_codec_width;
        m_height = m_codec_height;
    }

    m_fps = av_q2d(stream->avg_frame_rate);
    if (m_fps <= 0) m_fps = av_q2d(stream->r_frame_rate);
    if (m_fps <= 0) m_fps = m_cfg.capture_fps;

    m_codec = avcodec_find_decoder(params->codec_id);
    if (!m_codec) {
        LOGE("Codec not found");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Setup decoder (HW or SW)
// ---------------------------------------------------------------------------
bool VideoDecoder::setup_decoder() {
    AVCodecParameters* params = m_fmt_ctx->streams[m_video_stream_idx]->codecpar;
    m_codec_ctx = avcodec_alloc_context3(m_codec);
    avcodec_parameters_to_context(m_codec_ctx, params);

    if (m_hw_type == HW_VAAPI) {
        return init_hw_decoder();
    }

    // SW decoder: use multi-threading
    m_codec_ctx->thread_count = 0;  // auto
    m_codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    int ret = avcodec_open2(m_codec_ctx, m_codec, nullptr);
    if (ret < 0) {
        LOGE("avcodec_open2 failed");
        return false;
    }
    return true;
}

bool VideoDecoder::init_hw_decoder() {
    AVHWDeviceType hw_type = (m_hw_type == HW_VAAPI)
        ? AV_HWDEVICE_TYPE_VAAPI
        : AV_HWDEVICE_TYPE_CUDA;

    int ret = av_hwdevice_ctx_create(&m_hw_ctx, hw_type,
        m_cfg.hw_device.empty() ? nullptr : m_cfg.hw_device.c_str(),
        nullptr, 0);
    if (ret < 0) {
        LOGW("HW device create failed, falling back to SW");
        m_hw_type = HW_NONE;
        m_codec_ctx->thread_count = 0;
        return avcodec_open2(m_codec_ctx, m_codec, nullptr) >= 0;
    }

    m_codec_ctx->hw_device_ctx = av_buffer_ref(m_hw_ctx);

    // Find HW pixel format for this codec
    const AVCodecHWConfig* hw_cfg = nullptr;
    for (int i = 0; (hw_cfg = avcodec_get_hw_config(m_codec, i)) != nullptr; i++) {
        if (hw_cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            hw_cfg->device_type == hw_type) {
            m_src_pix_fmt = hw_cfg->pix_fmt;
            break;
        }
    }

    ret = avcodec_open2(m_codec_ctx, m_codec, nullptr);
    if (ret < 0) {
        LOGW("avcodec_open2 (HW) failed, fallback SW");
        av_buffer_unref(&m_hw_ctx);
        m_hw_type = HW_NONE;
        m_codec_ctx->hw_device_ctx = nullptr;
        m_codec_ctx->thread_count = 0;
        return avcodec_open2(m_codec_ctx, m_codec, nullptr) >= 0;
    }

    LOGI("Hardware decoding enabled (%s)", m_cfg.hw_accel.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// SwsContext for pixel format conversion
// ---------------------------------------------------------------------------
bool VideoDecoder::setup_sws() {
    AVPixelFormat out_fmt = m_cfg.decoder_output_yuv
        ? AV_PIX_FMT_YUV420P : AV_PIX_FMT_BGR24;

    m_bgr_frame->format = out_fmt;
    m_bgr_frame->width  = m_codec_width;
    m_bgr_frame->height = m_codec_height;
    av_frame_get_buffer(m_bgr_frame, 0);

    m_sws_ctx = sws_getContext(
        m_codec_width, m_codec_height, m_src_pix_fmt,
        m_codec_width, m_codec_height, out_fmt,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    return m_sws_ctx != nullptr;
}

// ---------------------------------------------------------------------------
// Public read: returns one PTS-annotated frame
// ---------------------------------------------------------------------------
bool VideoDecoder::read(DecodedFrame& out) {
    if (!m_is_open) return false;

    // Return buffered frame if available (jitter smoothing)
    if (!m_frame_buf.empty()) {
        out = std::move(m_frame_buf.front());
        m_frame_buf.pop_front();
        return true;
    }

    while (true) {
        // Decode next packet
        bool got_frame = decode_packet();
        if (!got_frame) return false;

        // Try to receive a decoded frame
        if (receive_and_convert(out)) {
            if (fix_jitter(out)) {
                out.frame_idx = m_display_idx++;
                return true;
            } 
            // jitter fix decided to drop/skip this frame, loop again
        } 
        // EAGAIN: need more packets, loop
    }
}

// ---------------------------------------------------------------------------
// Decode one AVPacket, returns true if data was sent to codec
// ---------------------------------------------------------------------------
bool VideoDecoder::decode_packet() {
    while (true) {
        int ret = av_read_frame(m_fmt_ctx, m_pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                if (m_eof) return false;
                m_eof = true;
                // Flush decoder
                avcodec_send_packet(m_codec_ctx, nullptr);
                return true;
            }
            m_consecutive_errors++;
            if (m_consecutive_errors > m_cfg.max_consecutive_drops * 2) return false;
            continue;
        }
        m_consecutive_errors = 0;

        if (m_pkt->stream_index != m_video_stream_idx) {
            av_packet_unref(m_pkt);
            continue;
        }

        ret = avcodec_send_packet(m_codec_ctx, m_pkt);
        av_packet_unref(m_pkt);
        if (ret < 0) {
            m_consecutive_drops++;
            if (m_consecutive_drops > m_cfg.max_consecutive_drops) {
                LOGE("Too many decode errors, skipping packet");
                return false;
            }
            continue;
        }
        m_consecutive_drops = 0;
        return true;
    }
}

// ---------------------------------------------------------------------------
// Receive decoded frame + convert to output format + extract PTS
// ---------------------------------------------------------------------------
bool VideoDecoder::receive_and_convert(DecodedFrame& out) {
    // HW decode: get HW surface first, then copy to CPU
    AVFrame* src = (m_hw_type != HW_NONE) ? m_hw_frame : m_frame;

    int ret = avcodec_receive_frame(m_codec_ctx, src);
    if (ret == AVERROR(EAGAIN)) return false;   // need more packets
    if (ret == AVERROR_EOF) return false;
    if (ret < 0) return false;

    m_decode_idx++;

    // If HW, transfer from GPU to CPU
    AVFrame* cpu_frame = src;
    if (m_hw_type != HW_NONE && src->format == (int)m_src_pix_fmt) {
        ret = av_hwframe_transfer_data(m_frame, src, 0);
        if (ret < 0) {
            LOGE("HW→CPU transfer failed");
            av_frame_unref(src);
            return false;
        }
        av_frame_unref(src);
        cpu_frame = m_frame;
    }

    // Extract PTS
    out.pts        = cpu_frame->pts;
    out.pts_sec    = pts_to_sec(cpu_frame->pts);
    out.duration   = cpu_frame->duration;
    out.is_keyframe = (cpu_frame->flags & AV_FRAME_FLAG_KEY) != 0;

    if (out.duration <= 0 && m_fps > 0)
        out.duration = (int64_t)(m_time_base.den / (m_fps * m_time_base.num) + 0.5);

    if (out.pts == AV_NOPTS_VALUE) {
        out.pts = (int64_t)(m_decode_idx * (m_time_base.den / (m_fps * m_time_base.num)));
        out.pts_sec = pts_to_sec(out.pts);
    }

    // Convert to output format (BGR or YUV)
    if (m_cfg.decoder_output_yuv) {
        // Direct copy if already YUV420P
        if (cpu_frame->format == AV_PIX_FMT_YUV420P &&
            cpu_frame->width == m_codec_width && cpu_frame->height == m_codec_height) {
            av_frame_copy(m_bgr_frame, cpu_frame);
        } else {
            sws_scale(m_sws_ctx, cpu_frame->data, cpu_frame->linesize,
                      0, m_codec_height,
                      m_bgr_frame->data, m_bgr_frame->linesize);
        }
    } else {
        sws_scale(m_sws_ctx, cpu_frame->data, cpu_frame->linesize,
                  0, m_codec_height,
                  m_bgr_frame->data, m_bgr_frame->linesize);
    }

    av_frame_unref(m_frame);
    if (m_hw_type != HW_NONE) av_frame_unref(m_hw_frame);

    // Zero-copy wrap into cv::Mat + clone for safety
    out.data = cv::Mat(m_codec_height, m_codec_width, CV_8UC3,
                       m_bgr_frame->data[0],
                       m_bgr_frame->linesize[0]).clone();

    // Apply rotation
    if (m_rotation == 90)
        cv::rotate(out.data, out.data, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (m_rotation == 180)
        cv::rotate(out.data, out.data, cv::ROTATE_180);
    else if (m_rotation == 270)
        cv::rotate(out.data, out.data, cv::ROTATE_90_CLOCKWISE);

    out.valid = true;
    return true;
}

// ---------------------------------------------------------------------------
// PTS → seconds conversion
// ---------------------------------------------------------------------------
double VideoDecoder::pts_to_sec(int64_t pts) const {
    if (pts == AV_NOPTS_VALUE) return -1.0;
    return (double)pts * m_time_base.num / m_time_base.den;
}

// ---------------------------------------------------------------------------
// Jitter detection and frame-rate smoothing
// ---------------------------------------------------------------------------
bool VideoDecoder::fix_jitter(DecodedFrame& frame) {
    if (m_last_pts == AV_NOPTS_VALUE) {
        m_last_pts = frame.pts;
        m_expected_next_pts = frame.pts + frame.duration;
        return true;
    }

    int64_t expected_duration = frame.duration > 0
        ? frame.duration
        : (int64_t)(m_time_base.den / (m_fps * m_time_base.num));

    double jitter_ms = std::abs(frame.pts - m_expected_next_pts)
        * m_time_base.num * 1000.0 / m_time_base.den;

    if (jitter_ms <= m_cfg.max_jitter_ms) {
        // Normal: within jitter tolerance
        m_last_pts = frame.pts;
        m_expected_next_pts = frame.pts + expected_duration;
        return true;
    }

    if (frame.pts < m_expected_next_pts && jitter_ms > m_cfg.max_jitter_ms) {
        // Frame arrived too early or is a duplicate — drop it
        LOGW("Dropping early/duplicate frame: PTS jitter=%.2fms", jitter_ms);
        return false;
    }

    // Frame arrived late (gap) — insert padding frame(s) into buffer
    int64_t gap = frame.pts - m_expected_next_pts;
    int fill_count = std::max(0, (int)(gap / expected_duration));
    fill_count = std::min(fill_count, (int)m_cfg.max_consecutive_drops);

    if (fill_count > 0) {
        LOGI("PTS gap %.2fms, inserting %d fill frame(s)",
             gap * m_time_base.num * 1000.0 / m_time_base.den, fill_count);
    }

    for (int i = 0; i < fill_count && m_frame_buf.size() < kMaxBufSize; i++) {
        DecodedFrame fill;
        fill.data      = frame.data.clone();
        fill.pts       = m_expected_next_pts;
        fill.pts_sec   = pts_to_sec(fill.pts);
        fill.duration  = expected_duration;
        fill.is_keyframe = false;
        fill.valid     = true;
        m_frame_buf.push_back(std::move(fill));
        m_last_pts = fill.pts;
        m_expected_next_pts = fill.pts + expected_duration;
    }

    // Now accept the current frame
    m_last_pts = frame.pts;
    m_expected_next_pts = frame.pts + expected_duration;
    return true;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
void VideoDecoder::close() {
    m_is_open = false;
    m_eof = false;
    m_frame_buf.clear();

    if (m_sws_ctx) { sws_freeContext(m_sws_ctx); m_sws_ctx = nullptr; }
    if (m_codec_ctx) { avcodec_free_context(&m_codec_ctx); }
    if (m_fmt_ctx) { avformat_close_input(&m_fmt_ctx); }
    if (m_hw_ctx) { av_buffer_unref(&m_hw_ctx); }
    if (m_pkt) { av_packet_free(&m_pkt); }
    if (m_frame) { av_frame_free(&m_frame); }
    if (m_hw_frame) { av_frame_free(&m_hw_frame); }
    if (m_bgr_frame) { av_frame_free(&m_bgr_frame); }

    LOGI("Closed. Frames: decoded=%u displayed=%u", m_decode_idx, m_display_idx);
}
