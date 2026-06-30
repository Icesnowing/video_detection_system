#include "video_decoder.h"
#include <iostream>
#include <cstring>

VideoDecoder::VideoDecoder(const Config& cfg) : m_cfg(cfg) {
    m_pkt = av_packet_alloc();
    m_frame = av_frame_alloc();
    m_bgr_frame = av_frame_alloc();
}

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open() {
    if (!open_input()) return false;
    if (!find_video_stream()) return false;
    if (!setup_decoder()) return false;

    m_is_open = true;
    std::cout << "[Decoder] Opened: " << m_width << "x" << m_height
              << " @ " << m_fps << "fps" << std::endl;
    return true;
}

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
    }

    const AVInputFormat* input_fmt = nullptr;
    if (!fmt_name.empty()) {
        input_fmt = av_find_input_format(fmt_name.c_str());
    }

    int ret = avformat_open_input(&m_fmt_ctx, src.c_str(), input_fmt, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[Decoder] avformat_open_input failed: " << errbuf << std::endl;
        return false;
    }

    ret = avformat_find_stream_info(m_fmt_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "[Decoder] avformat_find_stream_info failed" << std::endl;
        return false;
    }

    return true;
}

bool VideoDecoder::find_video_stream() {
    m_video_stream_idx = -1;

    for (unsigned i = 0; i < m_fmt_ctx->nb_streams; i++) {
        if (m_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_video_stream_idx = i;
            break;
        }
    }

    if (m_video_stream_idx < 0) {
        std::cerr << "[Decoder] No video stream found" << std::endl;
        return false;
    }

    AVStream* stream = m_fmt_ctx->streams[m_video_stream_idx];
    AVCodecParameters* params = stream->codecpar;

    m_codec_width = params->width;
    m_codec_height = params->height;

    if (m_codec_width <= 0 || m_codec_height <= 0) {
        m_codec_width = m_cfg.capture_width;
        m_codec_height = m_cfg.capture_height;
    }

    // Read display rotation metadata (common in phone videos)
    for (int i = 0; i < params->nb_coded_side_data; i++) {
        const auto& sd = params->coded_side_data[i];
        if (sd.type == AV_PKT_DATA_DISPLAYMATRIX && sd.size >= 9 * 4) {
            double rotation = av_display_rotation_get((const int32_t*)sd.data);
            int rot_deg = (int)(rotation + 360.0) % 360;
            if (rot_deg == 90 || rot_deg == 270) {
                m_rotation = rot_deg;
                std::cout << "[Decoder] Detected " << rot_deg << "° rotation" << std::endl;
            } else if (rot_deg == 180) {
                m_rotation = 180;
                std::cout << "[Decoder] Detected 180° rotation" << std::endl;
            }
            break;
        }
    }

    // Set display dimensions (swap for 90/270 rotation)
    if (m_rotation == 90 || m_rotation == 270) {
        m_width = m_codec_height;
        m_height = m_codec_width;
    } else {
        m_width = m_codec_width;
        m_height = m_codec_height;
    }

    m_fps = av_q2d(stream->avg_frame_rate);
    if (m_fps <= 0) m_fps = m_cfg.capture_fps;

    // Select codec
    m_codec = avcodec_find_decoder(params->codec_id);
    if (!m_codec) {
        std::cerr << "[Decoder] Codec not found" << std::endl;
        return false;
    }

    return true;
}

bool VideoDecoder::setup_decoder() {
    AVCodecParameters* params = m_fmt_ctx->streams[m_video_stream_idx]->codecpar;

    m_codec_ctx = avcodec_alloc_context3(m_codec);
    avcodec_parameters_to_context(m_codec_ctx, params);

    // Hardware acceleration (if available)
    if (m_cfg.hw_accel == "vaapi") {
        m_codec_ctx->hw_device_ctx = nullptr;  // Would need av_hwdevice_ctx_create
        // Simplified: fall through to software decode
    }

    int ret = avcodec_open2(m_codec_ctx, m_codec, nullptr);
    if (ret < 0) {
        std::cerr << "[Decoder] avcodec_open2 failed" << std::endl;
        return false;
    }

    // BGR frame setup (use codec dimensions, rotation applied later)
    m_bgr_frame->format = AV_PIX_FMT_BGR24;
    m_bgr_frame->width = m_codec_width;
    m_bgr_frame->height = m_codec_height;
    av_frame_get_buffer(m_bgr_frame, 0);

    // SwsContext for format conversion
    m_sws_ctx = sws_getContext(
        m_codec_width, m_codec_height, (AVPixelFormat)params->format,
        m_codec_width, m_codec_height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    return true;
}

bool VideoDecoder::read(cv::Mat& frame) {
    if (!m_is_open) return false;

    while (true) {
        int ret = av_read_frame(m_fmt_ctx, m_pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) return false;
            continue;
        }

        if (m_pkt->stream_index != m_video_stream_idx) {
            av_packet_unref(m_pkt);
            continue;
        }

        ret = avcodec_send_packet(m_codec_ctx, m_pkt);
        av_packet_unref(m_pkt);

        if (ret < 0) continue;

        ret = avcodec_receive_frame(m_codec_ctx, m_frame);
        if (ret == AVERROR(EAGAIN)) continue;
        if (ret < 0) return false;

        // Convert to BGR (using SwsContext)
        // If the decoded format is already BGR24, copy directly
        if (m_frame->format == AV_PIX_FMT_BGR24) {
            av_frame_copy(m_bgr_frame, m_frame);
        } else {
            sws_scale(m_sws_ctx,
                      m_frame->data, m_frame->linesize, 0, m_codec_height,
                      m_bgr_frame->data, m_bgr_frame->linesize);
        }

        av_frame_unref(m_frame);

        // Zero-copy wrap into cv::Mat (codec dimensions, rotation applied next)
        frame = cv::Mat(m_codec_height, m_codec_width, CV_8UC3,
                        m_bgr_frame->data[0],
                        m_bgr_frame->linesize[0]).clone();

        // Apply rotation if needed (phone videos)
        if (m_rotation == 90) {
            cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        } else if (m_rotation == 180) {
            cv::rotate(frame, frame, cv::ROTATE_180);
        } else if (m_rotation == 270) {
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        }

        m_frame_count++;
        return true;
    }
}

void VideoDecoder::close() {
    m_is_open = false;

    if (m_sws_ctx) {
        sws_freeContext(m_sws_ctx);
        m_sws_ctx = nullptr;
    }
    if (m_codec_ctx) {
        avcodec_free_context(&m_codec_ctx);
    }
    if (m_fmt_ctx) {
        avformat_close_input(&m_fmt_ctx);
    }
    if (m_pkt) {
        av_packet_free(&m_pkt);
    }
    if (m_frame) {
        av_frame_free(&m_frame);
    }
    if (m_bgr_frame) {
        av_frame_free(&m_bgr_frame);
    }

    std::cout << "[Decoder] Closed. Total frames: " << m_frame_count << std::endl;
}
