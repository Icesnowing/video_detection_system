#include "common.h"
#include "video_decoder.h"
#include "yuv_decoder.h"
#include "preprocessor.h"
#include "inference_onnx.h"
#include "postprocessor.h"
#include "video_encoder.h"
#include "yuv_encoder.h"
#include "trace_log.h"

#include <iomanip>
#include <sstream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cmath>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

class DetectionPipeline {
public:
    DetectionPipeline(const Config& cfg)
        : m_cfg(cfg),
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

        //m_inference.warmup(5);

        int out_w = (m_cfg.source_type == "yuv") ? m_yuv_decoder.width() : m_decoder.width();
        int out_h = (m_cfg.source_type == "yuv") ? m_yuv_decoder.height() : m_decoder.height();

        bool is_yuv_out = m_cfg.output_path.size() >= 4 &&
                          m_cfg.output_path.substr(m_cfg.output_path.size() - 4) == ".yuv";
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
        if (!setup()) return;

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        using Clock = std::chrono::steady_clock;
        auto elapsed_ms = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };

        auto t_start = Clock::now();
        auto last_log = t_start;
        auto last_sample_time = t_start;
        double total_inf_time = 0.0;
        int frame_idx = 0;
        std::vector<Detection> last_detections;
        int sampled_count = 0;

        trace_log_raw("[Latency] unit=ms; postprocess_ms includes NMS and box drawing");
        trace_log_raw("[Latency] frame pts_sec decode_ms preprocess_ms infer_ms postprocess_ms encode_ms total_ms detect");

        while (g_running) {
            // 1. Decode (with PTS timestamps)
            auto t_stage = Clock::now();
            DecodedFrame dec_frame;
            bool ok;
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
            double decode_ms = elapsed_ms(t_stage);
            if (!ok) {
                LOGI("End of stream.");
                break;
            }

            cv::Mat& frame = dec_frame.data;

            // 2. Decide whether to run detection (frame sampling)
            bool run_detection = false;
            if (m_cfg.sample_mode == "none") {
                run_detection = true;
            } else if (m_cfg.sample_mode == "keyframe") {
                run_detection = dec_frame.is_keyframe;
            } else if (m_cfg.sample_mode == "skip") {
                run_detection = (m_cfg.sample_skip_n <= 1 || frame_idx % m_cfg.sample_skip_n == 0);
            } else if (m_cfg.sample_mode == "interval") {
                auto now = Clock::now();
                double since_last = std::chrono::duration<double>(now - last_sample_time).count();
                if (since_last >= m_cfg.sample_interval_sec) {
                    run_detection = true;
                    last_sample_time = now;
                }
            }

            std::vector<Detection> detections;
            double preprocess_ms = 0.0;
            double infer_ms = 0.0;
            double postprocess_ms = 0.0;

            if (run_detection) {
                // 3. Preprocess
                t_stage = Clock::now();
                auto preproc = m_preprocessor.process(frame);
                preprocess_ms = elapsed_ms(t_stage);

                // 4. Inference
                t_stage = Clock::now();
                std::vector<float> output_data;
                std::vector<int64_t> output_shape;
                m_inference.infer(preproc.tensor, preproc.shape,
                                  output_data, output_shape);
                infer_ms = elapsed_ms(t_stage);
                total_inf_time += infer_ms;

                // 5. Postprocess + box drawing
                t_stage = Clock::now();
                detections = m_postprocessor.process(
                    output_data, output_shape,
                    frame.cols, frame.rows,
                    preproc.ratio_w, preproc.ratio_h,
                    preproc.pad_left, preproc.pad_top);

                last_detections = detections;
                sampled_count++;
                // Stamp detections with frame PTS
                for (auto& d : detections) {
                    d.pts = dec_frame.pts;
                    d.pts_sec = dec_frame.pts_sec;
                    d.frame_idx = dec_frame.frame_idx;
                }
                m_renderer.draw(frame, detections);
                postprocess_ms = elapsed_ms(t_stage);
            } else {
                detections = last_detections;
            }

            // 6. Encode (with PTS for timing)
            t_stage = Clock::now();
            bool is_yuv_out = m_cfg.output_path.size() >= 4 &&
                              m_cfg.output_path.substr(m_cfg.output_path.size() - 4) == ".yuv";
            if (is_yuv_out) {
                m_yuv_encoder.write(frame);
            } else {
                m_encoder.write(frame, dec_frame.pts_sec, dec_frame.frame_idx);
            }
            double encode_ms = elapsed_ms(t_stage);
            double total_ms = decode_ms + preprocess_ms + infer_ms + postprocess_ms + encode_ms;

            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3);
                oss << "[Latency] frame=" << frame_idx
                    << " pts_sec=" << dec_frame.pts_sec
                    << " decode_ms=" << decode_ms
                    << " preprocess_ms=" << preprocess_ms
                    << " infer_ms=" << infer_ms
                    << " postprocess_ms=" << postprocess_ms
                    << " encode_ms=" << encode_ms
                    << " total_ms=" << total_ms
                    << " detect=" << (run_detection ? 1 : 0);
                trace_log_raw(oss.str());
            }

            // 7. Display
            if (m_cfg.show_window) {
                cv::imshow(m_cfg.window_name, frame);
                if (cv::waitKey(1) == 'q') break;
            }

            frame_idx++;

            // Logging every 1 seconds
            auto now = Clock::now();
            double elapsed = std::chrono::duration<double>(now - last_log).count();
            if (elapsed >= 1.0) {
                double total_elapsed = std::chrono::duration<double>(now - t_start).count();
                double fps = frame_idx / total_elapsed;
                double avg_lat = total_inf_time / frame_idx;

                LOGI("Frame %d | FPS: %.2f | Inf: %.2fms | Det: %zu | PTS: %.3fs",
                     frame_idx, fps, avg_lat, detections.size(), dec_frame.pts_sec);

                last_log = now;
            }
        }

        shutdown();
    }

    void shutdown() {
        m_decoder.close();
        m_yuv_decoder.close();
        m_encoder.close();
        m_yuv_encoder.close();
        if (m_cfg.show_window) cv::destroyAllWindows();
        LOGI("Shutdown complete.");
    }

private:
    Config m_cfg;
    VideoDecoder m_decoder;
    YuvDecoder m_yuv_decoder;
    Preprocessor m_preprocessor;
    InferenceEngine m_inference;
    PostProcessor m_postprocessor;
    Renderer m_renderer;
    VideoEncoder m_encoder;
    YuvEncoder m_yuv_encoder;
};
