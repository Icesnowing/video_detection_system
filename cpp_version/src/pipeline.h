#include "common.h"
#include "video_decoder.h"
#include "yuv_decoder.h"
#include "preprocessor.h"
#include "inference_onnx.h"
#include "postprocessor.h"
#include "video_encoder.h"
#include "yuv_encoder.h"

#include <iomanip>
#include <iostream>
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
        std::cout << "[Pipeline] Initializing..." << std::endl;

        if (m_cfg.source_type == "yuv") {
            if (!m_yuv_decoder.open()) {
                std::cerr << "[Pipeline] ERROR: Failed to open YUV decoder." << std::endl;
                return false;
            }
        } else {
            if (!m_decoder.open()) {
                std::cerr << "[Pipeline] ERROR: Failed to open decoder." << std::endl;
                return false;
            }
        }

        if (!m_inference.load()) {
            std::cerr << "[Pipeline] ERROR: Failed to load engine." << std::endl;
            return false;
        }

        //m_inference.warmup(5);

        int out_w = (m_cfg.source_type == "yuv") ? m_yuv_decoder.width() : m_decoder.width();
        int out_h = (m_cfg.source_type == "yuv") ? m_yuv_decoder.height() : m_decoder.height();

        bool is_yuv_out = m_cfg.output_path.size() >= 4 &&
                          m_cfg.output_path.substr(m_cfg.output_path.size() - 4) == ".yuv";
        if (is_yuv_out) {
            if (!m_yuv_encoder.open(out_w, out_h)) {
                std::cerr << "[Pipeline] WARNING: YUV encoder failed." << std::endl;
            }
        } else {
            if (!m_encoder.open(out_w, out_h)) {
                std::cerr << "[Pipeline] WARNING: Encoder failed." << std::endl;
            }
        }

        std::cout << "[Pipeline] Initialization complete." << std::endl;
        return true;
    }

    void run() {
        if (!setup()) return;

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        auto t_start = std::chrono::high_resolution_clock::now();
        auto last_log = t_start;
        double total_inf_time = 0.0;
        int frame_idx = 0;

        while (g_running) {
            // 1. Decode (with PTS timestamps)
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
            if (!ok) {
                std::cout << "[Pipeline] End of stream." << std::endl;
                break;
            }

            cv::Mat& frame = dec_frame.data;

            // 2. Preprocess
            auto preproc = m_preprocessor.process(frame);

            // 3. Inference
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<float> output_data;
            std::vector<int64_t> output_shape;
            m_inference.infer(preproc.tensor, preproc.shape,
                              output_data, output_shape);

            auto t1 = std::chrono::high_resolution_clock::now();
            double latency = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_inf_time += latency;

            // 4. Postprocess
            auto detections = m_postprocessor.process(
                output_data, output_shape,
                frame.cols, frame.rows,
                preproc.ratio_w, preproc.ratio_h,
                preproc.pad_left, preproc.pad_top);

            // Stamp detections with frame PTS
            for (auto& d : detections) {
                d.pts = dec_frame.pts;
                d.pts_sec = dec_frame.pts_sec;
                d.frame_idx = dec_frame.frame_idx;
            }

            // 5. Render
            m_renderer.draw(frame, detections);

            // 6. Encode (with PTS for timing)
            bool is_yuv_out = m_cfg.output_path.size() >= 4 &&
                              m_cfg.output_path.substr(m_cfg.output_path.size() - 4) == ".yuv";
            if (is_yuv_out) {
                m_yuv_encoder.write(frame);
            } else {
                m_encoder.write(frame, dec_frame.pts_sec, dec_frame.frame_idx);
            }

            // 7. Display
            if (m_cfg.show_window) {
                cv::imshow(m_cfg.window_name, frame);
                if (cv::waitKey(1) == 'q') break;
            }

            frame_idx++;

            // Logging every 1 seconds
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_log).count();
            if (elapsed >= 1.0) {
                double total_elapsed = std::chrono::duration<double>(now - t_start).count();
                double fps = frame_idx / total_elapsed;
                double avg_lat = total_inf_time / frame_idx;

                std::cout << "[Pipeline] Frame " << frame_idx
                          << " | FPS: " << fps
                          << " | Inf: " << avg_lat << "ms"
                          << " | Det: " << detections.size()
                          << " | PTS: " << dec_frame.pts_sec << "s" << std::endl;

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
        std::cout << "[Pipeline] Shutdown complete." << std::endl;
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
