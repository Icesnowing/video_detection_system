#pragma once

#include "common.h"
#include <onnxruntime/onnxruntime_cxx_api.h>

/**
 * ONNX Runtime inference engine (CPU-optimized, replaces TensorRT).
 *
 * Loads a YOLOv11 ONNX model, binds input/output tensors,
 * and runs inference with configurable execution providers.
 */
class InferenceEngine {
public:
    InferenceEngine(const Config& cfg);
    ~InferenceEngine();

    bool load();
    bool infer(const std::vector<float>& input_data,
               const std::vector<int64_t>& input_shape,
               std::vector<float>& output_data,
               std::vector<int64_t>& output_shape);

    void warmup(int num_runs = 5);
    void benchmark(int num_runs = 100);

private:
    Config m_cfg;

    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::SessionOptions> m_sess_opts;
    std::unique_ptr<Ort::Session> m_session;

    // Memory info for CPU allocations
    std::unique_ptr<Ort::MemoryInfo> m_mem_info;

    // Input/output metadata
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
    std::vector<int64_t> m_input_shape;
    std::vector<int64_t> m_output_shape;

    bool m_loaded = false;
};
