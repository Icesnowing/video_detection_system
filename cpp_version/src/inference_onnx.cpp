#include "inference_onnx.h"
#include <iostream>
#include <chrono>
#include <algorithm>

InferenceEngine::InferenceEngine(const Config& cfg) : m_cfg(cfg) {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VideoDetection");
    m_sess_opts = std::make_unique<Ort::SessionOptions>();

    // Graph optimization
    m_sess_opts->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Thread settings
    if (m_cfg.intra_op_threads > 0) {
        m_sess_opts->SetIntraOpNumThreads(m_cfg.intra_op_threads);
    }
    if (m_cfg.inter_op_threads > 0) {
        m_sess_opts->SetInterOpNumThreads(m_cfg.inter_op_threads);
    }

    if (m_cfg.enable_profiling) {
        m_sess_opts->EnableProfiling("ort_profile.json");
    }

    m_mem_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::load() {
    try {
        m_session = std::make_unique<Ort::Session>(
            *m_env, m_cfg.onnx_path.c_str(), *m_sess_opts);

        // Get input info
        size_t num_inputs = m_session->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::AllocatedStringPtr name = m_session->GetInputNameAllocated(i, allocator);
            m_input_names.push_back(name.release());

            Ort::TypeInfo type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            m_input_shape = tensor_info.GetShape();
        }

        // Get output info
        size_t num_outputs = m_session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::AllocatedStringPtr name = m_session->GetOutputNameAllocated(i, allocator);
            m_output_names.push_back(name.release());

            Ort::TypeInfo type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            m_output_shape = tensor_info.GetShape();
        }

        std::cout << "[Inference] Model loaded: " << m_cfg.onnx_path << std::endl;
        std::cout << "  Input shape: {";
        for (size_t i = 0; i < m_input_shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << m_input_shape[i];
        }
        std::cout << "}" << std::endl;

        m_loaded = true;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[Inference] ERROR: " << e.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::infer(
    const std::vector<float>& input_data,
    const std::vector<int64_t>& input_shape,
    std::vector<float>& output_data,
    std::vector<int64_t>& output_shape) {

    if (!m_loaded) return false;

    try {
        // Create input tensor from data
        size_t input_size = 1;
        for (auto d : input_shape) input_size *= d;

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *m_mem_info,
            const_cast<float*>(input_data.data()),
            input_size,
            input_shape.data(),
            input_shape.size()
        );

        // Run inference
        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            m_input_names.data(),
            &input_tensor,
            1,
            m_output_names.data(),
            m_output_names.size()
        );

        // Copy output data back
        const auto& output = outputs.front();
        auto out_info = output.GetTensorTypeAndShapeInfo();
        output_shape = out_info.GetShape();

        size_t out_size = 1;
        for (auto d : output_shape) out_size *= d;

        output_data.resize(out_size);
        std::memcpy(output_data.data(),
                    output.GetTensorData<float>(),
                    out_size * sizeof(float));

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[Inference] Runtime error: " << e.what() << std::endl;
        return false;
    }
}

void InferenceEngine::warmup(int num_runs) {
    std::vector<int64_t> shape = {1, 3,
                                  (int64_t)m_cfg.input_height,
                                  (int64_t)m_cfg.input_width};
    size_t size = shape[0] * shape[1] * shape[2] * shape[3];
    std::vector<float> dummy(size, 0.5f);

    std::vector<float> out;
    std::vector<int64_t> out_shape;

    std::cout << "[Inference] Warming up (" << num_runs << " runs)..." << std::endl;
    for (int i = 0; i < num_runs; i++) {
        infer(dummy, shape, out, out_shape);
    }
    std::cout << "[Inference] Warmup complete." << std::endl;
}

void InferenceEngine::benchmark(int num_runs) {
    std::vector<int64_t> shape = {1, 3,
                                  (int64_t)m_cfg.input_height,
                                  (int64_t)m_cfg.input_width};
    size_t size = shape[0] * shape[1] * shape[2] * shape[3];
    std::vector<float> dummy(size, 0.5f);

    std::vector<float> out;
    std::vector<int64_t> out_shape;

    std::vector<double> latencies;
    for (int i = 0; i < num_runs; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        infer(dummy, shape, out, out_shape);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());
    double sum = 0;
    for (auto l : latencies) sum += l;
    double mean = sum / latencies.size();
    double median = latencies[latencies.size() / 2];

    std::cout << "[Inference] Benchmark (" << num_runs << " runs):" << std::endl;
    std::cout << "  Mean:   " << mean << " ms" << std::endl;
    std::cout << "  Median: " << median << " ms" << std::endl;
    std::cout << "  Min:    " << latencies.front() << " ms" << std::endl;
    std::cout << "  Max:    " << latencies.back() << " ms" << std::endl;
    std::cout << "  FPS:    " << (1000.0 / mean) << std::endl;
}
