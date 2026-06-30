#include "common.h"
#include "video_decoder.h"
#include "preprocessor.h"
#include "inference_onnx.h"
#include "postprocessor.h"
#include "video_encoder.h"

#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running = false;
}

class DetectionPipeline {
public:
    DetectionPipeline(const Config& cfg)
        : m_cfg(cfg),
          m_decoder(cfg),
          m_preprocessor(cfg.input_width, cfg.input_height),
          m_inference(cfg),
          m_postprocessor(cfg),
          m_renderer(cfg),
          m_encoder(cfg) {}

    bool setup() {
        std::cout << "[Pipeline] Initializing..." << std::endl;

        if (!m_decoder.open()) {
            std::cerr << "[Pipeline] ERROR: Failed to open decoder." << std::endl;
            return false;
        }

        if (!m_inference.load()) {
            std::cerr << "[Pipeline] ERROR: Failed to load engine." << std::endl;
            return false;
        }

        m_inference.warmup(5);

        if (!m_encoder.open(m_decoder.width(), m_decoder.height())) {
            std::cerr << "[Pipeline] WARNING: Encoder failed." << std::endl;
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
            // 1. Decode
            cv::Mat frame;
            if (!m_decoder.read(frame)) {
                std::cout << "[Pipeline] End of stream." << std::endl;
                break;
            }

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

            // 5. Render
            m_renderer.draw(frame, detections);

            // 6. Encode
            m_encoder.write(frame);

            // 7. Display
            if (m_cfg.show_window) {
                cv::imshow(m_cfg.window_name, frame);
                if (cv::waitKey(1) == 'q') break;
            }

            frame_idx++;

            // Logging every 2 seconds
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_log).count();
            if (elapsed >= 2.0) {
                double total_elapsed = std::chrono::duration<double>(now - t_start).count();
                double fps = frame_idx / total_elapsed;
                double avg_lat = total_inf_time / frame_idx;

                std::cout << "[Pipeline] Frame " << frame_idx
                          << " | FPS: " << fps
                          << " | Inf: " << avg_lat << "ms"
                          << " | Det: " << detections.size() << std::endl;

                last_log = now;
            }
        }

        shutdown();
    }

    void shutdown() {
        m_decoder.close();
        m_encoder.close();
        if (m_cfg.show_window) cv::destroyAllWindows();
        std::cout << "[Pipeline] Shutdown complete." << std::endl;
    }

private:
    Config m_cfg;
    VideoDecoder m_decoder;
    Preprocessor m_preprocessor;
    InferenceEngine m_inference;
    PostProcessor m_postprocessor;
    Renderer m_renderer;
    VideoEncoder m_encoder;
};
