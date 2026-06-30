#pragma once

#include "common.h"

/**
 * Post-processing: confidence filtering, coordinate decoding,
 * hand-written Non-Maximum Suppression, and box rendering.
 */
class PostProcessor {
public:
    PostProcessor(const Config& cfg);
    ~PostProcessor() = default;

    /**
     * Process raw YOLO output into a list of detections.
     * output_shape: expected {1, 84, N}
     */
    std::vector<Detection> process(
        const std::vector<float>& output,
        const std::vector<int64_t>& output_shape,
        int original_width, int original_height,
        float ratio_w, float ratio_h,
        int pad_left, int pad_top);

private:
    void decode_boxes(const float* bbox_raw, int num_boxes,
                      std::vector<cv::Rect>& boxes) const;

    void scale_boxes(std::vector<cv::Rect>& boxes,
                     int model_w, int model_h,
                     int orig_w, int orig_h,
                     float ratio_w, float ratio_h,
                     int pad_left, int pad_top) const;

    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float iou_threshold) const;

    static std::vector<std::string> get_default_classes();

    Config m_cfg;
    std::vector<std::string> m_class_names;
    std::vector<cv::Scalar> m_colors;
};

class Renderer {
public:
    Renderer(const Config& cfg);
    ~Renderer() = default;

    void draw(cv::Mat& frame, const std::vector<Detection>& detections);

private:
    bool m_draw_labels;
    int m_box_thickness;
    float m_font_scale;
    std::vector<cv::Scalar> m_colors;
};
