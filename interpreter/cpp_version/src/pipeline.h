#pragma once

#include "blocking_ring_queue.h"
#include "frame_packet.h"
#include "ordered_merge.h"
#include "common.h"
#include "video_decoder.h"
#include "yuv_decoder.h"
#include "preprocessor.h"
#include "inference_onnx.h"
#include "postprocessor.h"
#include "video_encoder.h"
#include "yuv_encoder.h"
#include "trace_log.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

class DetectionPipeline {
public:
    explicit DetectionPipeline(const Config& cfg)
        : m_cfg(cfg),
          m_q_decode_to_preproc(queue_capacity()),
          m_q_det_preproc_to_infer(queue_capacity()),
          m_q_infer_to_merge(queue_capacity()),
          m_q_skip_preproc_to_merge(queue_capacity()),
          m_q_post_to_encode(queue_capacity()),
          m_decoder(cfg),
          m_yuv_decoder(cfg),
          m_preprocessor(cfg.input_width, cfg.input_height),
          m_inference(cfg),
          m_postprocessor(cfg),
          m_renderer(cfg),
          m_encoder(cfg),
          m_yuv_encoder(cfg) {}

    bool setup() {
        LOGI("Initializing...");

        if (m_cfg.source_type == "yuv") {
            if (!m_yuv_decoder.open()) {
                LOGE("Failed to open YUV decoder.");
                return false;
            }
        } else {
            if (!m_decoder.open()) {
                LOGE("Failed to open decoder.");
                return false;
            }
        }

        if (!m_inference.load()) {
            LOGE("Failed to load engine.");
            return false;
        }

        int out_w = (m_cfg.source_type == "yuv") ? m_yuv_decoder.width() : m_decoder.width();
        int out_h = (m_cfg.source_type == "yuv") ? m_yuv_decoder.height() : m_decoder.height();

        bool is_yuv_out = is_yuv_output_path();
        if (is_yuv_out) {
            if (!m_yuv_encoder.open(out_w, out_h)) {
                LOGW("YUV encoder failed.");
            }
        } else {
            if (!m_encoder.open(out_w, out_h)) {
                LOGW("Encoder failed.");
            }
        }

        LOGI("Initialization complete.");
        return true;
    }

    void run() {
        g_running = true;
        if (!setup()) {
            return;
        }

        const auto t_begin = Clock::now();
        m_encoded_frames = 0;

        std::signal(SIGINT, signal_handler);
#ifndef _WIN32
        std::signal(SIGTERM, signal_handler);
#endif

        trace_log_raw("[Latency] unit=ms; postprocess_ms includes NMS and box drawing");
        trace_log_raw("[Latency] frame pts_sec decode_ms preprocess_ms infer_ms postprocess_ms encode_ms total_ms detect");

        m_total_inf_time_ns.store(0, std::memory_order_relaxed);

        std::thread t_decode(&DetectionPipeline::decode_worker, this);
        std::thread t_preproc(&DetectionPipeline::preproc_worker, this);
        std::thread t_infer(&DetectionPipeline::infer_worker, this);
        std::thread t_post(&DetectionPipeline::post_worker, this);
        std::thread t_encode(&DetectionPipeline::encode_worker, this);

        t_decode.join();
        t_preproc.join();
        t_infer.join();
        t_post.join();
        t_encode.join();

        const double elapsed_sec =
            std::chrono::duration<double>(Clock::now() - t_begin).count();
        const double fps = elapsed_sec > 0.0
                               ? static_cast<double>(m_encoded_frames) / elapsed_sec
                               : 0.0;
        LOGI("Stream finished. Elapsed: %.3fs | Frames: %d | FPS: %.2f",
             elapsed_sec, m_encoded_frames, fps);

        m_q_decode_to_preproc.close();
        m_q_det_preproc_to_infer.close();
        m_q_infer_to_merge.close();
        m_q_skip_preproc_to_merge.close();
        m_q_post_to_encode.close();

        shutdown();
    }

    void shutdown() {
        m_decoder.close();
        m_yuv_decoder.close();
        m_encoder.close();
        m_yuv_encoder.close();
        if (m_cfg.show_window) {
            cv::destroyAllWindows();
        }
        LOGI("Shutdown complete.");
    }

private:
    size_t queue_capacity() const {
        return m_cfg.pipeline_queue_capacity > 0
                   ? static_cast<size_t>(m_cfg.pipeline_queue_capacity)
                   : 2;
    }

    // q / Ctrl+C only clear g_running. Decode then pushes eos so frames already
    // queued still encode. Failures close every ring so a blocked push/pop wakes.
    void abort_pipeline() {
        g_running.store(false, std::memory_order_relaxed);
        m_q_decode_to_preproc.close();
        m_q_det_preproc_to_infer.close();
        m_q_infer_to_merge.close();
        m_q_skip_preproc_to_merge.close();
        m_q_post_to_encode.close();
    }

    void push_eos(BlockingRingQueue<FramePacket>& queue) {
        if (queue.is_closed()) {
            return;
        }
        queue.push(FramePacket::make_eos());
    }

    bool is_yuv_output_path() const {
        return m_cfg.output_path.size() >= 4 &&
               m_cfg.output_path.substr(m_cfg.output_path.size() - 4) == ".yuv";
    }

    using Clock = std::chrono::steady_clock;

    static double elapsed_ms(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    bool decide_run_detection(uint32_t frame_idx, const DecodedFrame& dec, double pts_sec,
                              double& last_sample_pts_sec) const {
        if (m_cfg.sample_mode == "none") {
            return true;
        }
        if (m_cfg.sample_mode == "keyframe") {
            return dec.is_keyframe;
        }
        if (m_cfg.sample_mode == "skip") {
            return m_cfg.sample_skip_n <= 1 ||
                   (frame_idx % static_cast<uint32_t>(m_cfg.sample_skip_n) == 0);
        }
        if (m_cfg.sample_mode == "interval") {
            if (pts_sec - last_sample_pts_sec >= m_cfg.sample_interval_sec) {
                last_sample_pts_sec = pts_sec;
                return true;
            }
            return false;
        }
        return true;
    }

    void decode_worker() {
        uint32_t frame_idx = 0;
        double last_sample_pts_sec = -m_cfg.sample_interval_sec;

        try {
            for (;;) {
                if (!g_running || m_q_decode_to_preproc.is_closed()) {
                    push_eos(m_q_decode_to_preproc);
                    break;
                }

                auto t_stage = Clock::now();
                DecodedFrame dec_frame;
                bool ok = false;

                if (m_cfg.source_type == "yuv") {
                    cv::Mat yuv_frame;
                    ok = m_yuv_decoder.read(yuv_frame);
                    if (ok) {
                        dec_frame.data = yuv_frame;
                        dec_frame.pts_sec = frame_idx * (1.0 / m_cfg.output_fps);
                        dec_frame.pts = frame_idx;
                        dec_frame.frame_idx = frame_idx;
                        dec_frame.valid = true;
                    }
                } else {
                    ok = m_decoder.read(dec_frame);
                }

                const double decode_ms = elapsed_ms(t_stage);

                if (!ok) {
                    LOGI("End of stream.");
                    push_eos(m_q_decode_to_preproc);
                    break;
                }

                FramePacket packet;
                packet.frame = dec_frame.data;
                packet.frame_idx = frame_idx;
                packet.pts = dec_frame.pts;
                packet.pts_sec = dec_frame.pts_sec;
                packet.duration = dec_frame.duration;
                packet.is_keyframe = dec_frame.is_keyframe;
                packet.run_detection =
                    decide_run_detection(frame_idx, dec_frame, dec_frame.pts_sec, last_sample_pts_sec);
                packet.decode_ms = decode_ms;

                m_q_decode_to_preproc.push(std::move(packet));
                ++frame_idx;
                if (m_q_decode_to_preproc.is_closed()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            LOGE("Decode worker: %s", e.what());
            abort_pipeline();
        }
    }

    void preproc_worker() {
        try {
            for (;;) {
                FramePacket packet;
                if (!m_q_decode_to_preproc.pop(packet)) {
                    m_q_det_preproc_to_infer.close();
                    m_q_skip_preproc_to_merge.close();
                    break;
                }
                if (packet.eos) {
                    push_eos(m_q_det_preproc_to_infer);
                    push_eos(m_q_skip_preproc_to_merge);
                    break;
                }

                if (packet.run_detection) {
                    auto t_stage = Clock::now();
                    packet.preproc = m_preprocessor.process(packet.frame);
                    packet.preprocess_ms = elapsed_ms(t_stage);
                    m_q_det_preproc_to_infer.push(std::move(packet));
                } else {
                    packet.preprocess_ms = 0.0;
                    m_q_skip_preproc_to_merge.push(std::move(packet));
                }

                if (m_q_det_preproc_to_infer.is_closed() || m_q_skip_preproc_to_merge.is_closed()) {
                    m_q_det_preproc_to_infer.close();
                    m_q_skip_preproc_to_merge.close();
                    break;
                }
            }
        } catch (const std::exception& e) {
            LOGE("Preproc worker: %s", e.what());
            abort_pipeline();
        }
    }

    void infer_worker() {
        try {
            for (;;) {
                FramePacket packet;
                if (!m_q_det_preproc_to_infer.pop(packet)) {
                    m_q_infer_to_merge.close();
                    break;
                }
                if (packet.eos) {
                    push_eos(m_q_infer_to_merge);
                    break;
                }

                auto t_stage = Clock::now();
                m_inference.infer(packet.preproc.tensor, packet.preproc.shape, packet.output_data,
                                  packet.output_shape);
                packet.infer_ms = elapsed_ms(t_stage);

                const auto ns = static_cast<uint64_t>(packet.infer_ms * 1e6);
                m_total_inf_time_ns.fetch_add(ns, std::memory_order_relaxed);

                m_q_infer_to_merge.push(std::move(packet));
                if (m_q_infer_to_merge.is_closed()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            LOGE("Infer worker: %s", e.what());
            abort_pipeline();
        }
    }

    void post_worker() {
        OrderedLaneMerger merger(m_q_skip_preproc_to_merge, m_q_infer_to_merge);
        std::vector<Detection> last_detections;

        try {
            for (;;) {
                FramePacket packet;
                if (!merger.pop_next(packet)) {
                    LOGE("Post worker: merge failed");
                    abort_pipeline();
                    break;
                }
                if (packet.eos) {
                    push_eos(m_q_post_to_encode);
                    break;
                }

                auto t_stage = Clock::now();
                if (packet.run_detection) {
                    packet.detections = m_postprocessor.process(
                        packet.output_data, packet.output_shape, packet.frame.cols, packet.frame.rows,
                        packet.preproc.ratio_w, packet.preproc.ratio_h, packet.preproc.pad_left,
                        packet.preproc.pad_top);

                    for (auto& d : packet.detections) {
                        d.pts = packet.pts;
                        d.pts_sec = packet.pts_sec;
                        d.frame_idx = static_cast<int>(packet.frame_idx);
                    }
                    m_renderer.draw(packet.frame, packet.detections);
                    last_detections = packet.detections;
                } else if (!last_detections.empty()) {
                    packet.detections = last_detections;
                    m_renderer.draw(packet.frame, packet.detections);
                } else {
                    packet.detections.clear();
                }
                packet.postprocess_ms = elapsed_ms(t_stage);

                m_q_post_to_encode.push(std::move(packet));
                if (m_q_post_to_encode.is_closed()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            LOGE("Post worker: %s", e.what());
            abort_pipeline();
        }
    }

    void encode_worker() {
        const auto t_start = Clock::now();
        auto last_log = t_start;

        try {
            for (;;) {
                FramePacket packet;
                if (!m_q_post_to_encode.pop(packet)) {
                    break;
                }
                if (packet.eos) {
                    break;
                }

                auto t_stage = Clock::now();
                if (is_yuv_output_path()) {
                    m_yuv_encoder.write(packet.frame);
                } else {
                    m_encoder.write(packet.frame, packet.pts_sec, packet.frame_idx);
                }
                packet.encode_ms = elapsed_ms(t_stage);

                const double total_ms = packet.decode_ms + packet.preprocess_ms + packet.infer_ms +
                                        packet.postprocess_ms + packet.encode_ms;

                {
                    LOGI("frame = %u, pts_sec = %.3fs, decode_ms = %.3fms, preprocess_ms = %.3fms, infer_ms = %.3fms, postprocess_ms = %.3fms, encode_ms = %.3fms, total_ms = %.3fms, detect = %d",
                         packet.frame_idx, packet.pts_sec, packet.decode_ms, packet.preprocess_ms, packet.infer_ms, packet.postprocess_ms, packet.encode_ms, total_ms, packet.run_detection ? 1 : 0);
                }

                if (m_cfg.show_window) {
                    cv::imshow(m_cfg.window_name, packet.frame);
                    if (cv::waitKey(1) == 'q') {
                        g_running = false;
                    }
                }

                ++m_encoded_frames;

                const auto now = Clock::now();
                const double since_log =
                    std::chrono::duration<double>(now - last_log).count();
                if (since_log >= 1.0) {
                    const double total_elapsed =
                        std::chrono::duration<double>(now - t_start).count();
                    const double fps =
                        m_encoded_frames > 0 ? m_encoded_frames / total_elapsed : 0.0;
                    const double total_inf_ms =
                        static_cast<double>(m_total_inf_time_ns.load(std::memory_order_relaxed)) /
                        1e6;
                    const double avg_inf =
                        m_encoded_frames > 0 ? total_inf_ms / m_encoded_frames : 0.0;

                    LOGI("Frame %u | FPS: %.2f | Inf: %.2fms | Det: %zu | PTS: %.3fs",
                         packet.frame_idx, fps, avg_inf, packet.detections.size(), packet.pts_sec);
                    last_log = now;
                }
            }
        } catch (const std::exception& e) {
            LOGE("Encode worker: %s", e.what());
            abort_pipeline();
        }
    }

    Config m_cfg;
    BlockingRingQueue<FramePacket> m_q_decode_to_preproc;
    BlockingRingQueue<FramePacket> m_q_det_preproc_to_infer;
    BlockingRingQueue<FramePacket> m_q_infer_to_merge;
    BlockingRingQueue<FramePacket> m_q_skip_preproc_to_merge;
    BlockingRingQueue<FramePacket> m_q_post_to_encode;

    VideoDecoder m_decoder;
    YuvDecoder m_yuv_decoder;
    Preprocessor m_preprocessor;
    InferenceEngine m_inference;
    PostProcessor m_postprocessor;
    Renderer m_renderer;
    VideoEncoder m_encoder;
    YuvEncoder m_yuv_encoder;

    std::atomic<uint64_t> m_total_inf_time_ns{0};
    int m_encoded_frames = 0;
};
