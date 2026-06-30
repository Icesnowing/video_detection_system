#include "postprocessor.h"
#include <algorithm>
#include <iostream>

std::vector<std::string> PostProcessor::get_default_classes() {
    return {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}

PostProcessor::PostProcessor(const Config& cfg) : m_cfg(cfg) {
    m_class_names = get_default_classes();

    // Generate stable colors
    m_colors.resize(m_class_names.size());
    for (size_t i = 0; i < m_class_names.size(); i++) {
        int r = (i * 113 + 67) % 256;
        int g = (i * 179 + 89) % 256;
        int b = (i * 241 + 97) % 256;
        m_colors[i] = cv::Scalar(b, g, r);  // BGR
    }
}

std::vector<Detection> PostProcessor::process(
    const std::vector<float>& output,
    const std::vector<int64_t>& output_shape,
    int original_width, int original_height,
    float ratio_w, float ratio_h,
    int pad_left, int pad_top) {

    // output_shape: {1, 84, N} -> N predictions, each with 4 bbox + 80 class scores
    if (output_shape.size() < 3) return {};

    int num_classes = output_shape[1] - 4;  // 80 for COCO
    int num_predictions = output_shape[2];

    // output is {1, 84, N}, transpose to {N, 84}
    // Each column j: output[j * 84 + i] for i in [0, 83]
    //   [0:4] = bbox (cx, cy, w, h)
    //   [4:84] = class scores

    std::vector<float> scores(num_predictions);
    std::vector<int> class_ids(num_predictions);
    std::vector<cv::Rect> boxes(num_predictions);
    std::vector<bool> valid(num_predictions, false);

    int step = 84;  // 4 + num_classes
    for (int i = 0; i < num_predictions; i++) {
        const float* row = &output[i * step];

        // Find max class score
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes; c++) {
            float score = row[4 + c];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }

        if (max_score < m_cfg.confidence_threshold) continue;

        // Decode bbox: cx, cy, w, h -> x1, y1, x2, y2
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        int x1 = (int)(cx - w / 2.0f);
        int y1 = (int)(cy - h / 2.0f);
        int x2 = (int)(cx + w / 2.0f);
        int y2 = (int)(cy + h / 2.0f);

        scores[i] = max_score;
        class_ids[i] = max_class;
        boxes[i] = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        valid[i] = true;
    }

    // Collect valid pre-NMS boxes
    std::vector<cv::Rect> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<int> valid_classes;
    for (int i = 0; i < num_predictions; i++) {
        if (valid[i]) {
            valid_boxes.push_back(boxes[i]);
            valid_scores.push_back(scores[i]);
            valid_classes.push_back(class_ids[i]);
        }
    }

    if (valid_boxes.empty()) return {};

    // Scale boxes from model input space to original frame
    scale_boxes(valid_boxes,
                m_cfg.input_width, m_cfg.input_height,
                original_width, original_height,
                ratio_w, ratio_h,
                pad_left, pad_top);

    // Per-class NMS
    std::vector<Detection> detections;
    int num_unique_classes = 80;
    for (int c = 0; c < num_unique_classes; c++) {
        std::vector<cv::Rect> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<int> cls_indices;

        for (size_t i = 0; i < valid_classes.size(); i++) {
            if (valid_classes[i] == c) {
                cls_boxes.push_back(valid_boxes[i]);
                cls_scores.push_back(valid_scores[i]);
                cls_indices.push_back(i);
            }
        }

        if (cls_boxes.empty()) continue;

        auto keep = nms(cls_boxes, cls_scores, m_cfg.nms_threshold);

        for (int idx : keep) {
            Detection det;
            det.bbox = cls_boxes[idx];
            det.confidence = cls_scores[idx];
            det.class_id = c;
            det.class_name = (c < (int)m_class_names.size())
                ? m_class_names[c] : ("class_" + std::to_string(c));
            detections.push_back(det);
        }
    }

    // Sort by confidence
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // Truncate to max_detections
    if ((int)detections.size() > m_cfg.max_detections) {
        detections.resize(m_cfg.max_detections);
    }

    return detections;
}

void PostProcessor::scale_boxes(
    std::vector<cv::Rect>& boxes,
    int model_w, int model_h,
    int orig_w, int orig_h,
    float ratio_w, float ratio_h,
    int pad_left, int pad_top) const {

    for (auto& box : boxes) {
        box.x -= pad_left;
        box.y -= pad_top;
        box.width += pad_left;   // adjust x2 coordinate
        box.height += pad_top;   // adjust y2 coordinate

        box.x = (int)(box.x / ratio_w);
        box.y = (int)(box.y / ratio_h);
        box.width = (int)((box.x + box.width) / ratio_w) - box.x;
        box.height = (int)((box.y + box.height) / ratio_h) - box.y;

        box.x = std::max(0, std::min(box.x, orig_w - 1));
        box.y = std::max(0, std::min(box.y, orig_h - 1));
        box.width = std::max(1, std::min(box.width, orig_w - box.x));
        box.height = std::max(1, std::min(box.height, orig_h - box.y));
    }
}

std::vector<int> PostProcessor::nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float iou_threshold) const {

    int n = boxes.size();
    std::vector<std::pair<float, int>> scored(n);
    for (int i = 0; i < n; i++) {
        scored[i] = {scores[i], i};
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; i++) {
        int idx = scored[i].second;
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (int j = i + 1; j < n; j++) {
            int idx2 = scored[j].second;
            if (suppressed[idx2]) continue;

            // Compute IoU
            const cv::Rect& a = boxes[idx];
            const cv::Rect& b = boxes[idx2];

            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.width, b.x + b.width);
            int y2 = std::min(a.y + a.height, b.y + b.height);

            int inter_w = std::max(0, x2 - x1);
            int inter_h = std::max(0, y2 - y1);
            float inter_area = (float)(inter_w * inter_h);

            float area_a = (float)(a.width * a.height);
            float area_b = (float)(b.width * b.height);
            float iou = inter_area / (area_a + area_b - inter_area + 1e-6f);

            if (iou > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }

    return keep;
}


// ---- Renderer ----

Renderer::Renderer(const Config& cfg)
    : m_draw_labels(cfg.draw_labels),
      m_box_thickness(cfg.box_thickness),
      m_font_scale(cfg.font_scale) {

    // Generate colors for 80 classes
    m_colors.resize(80);
    for (int i = 0; i < 80; i++) {
        int r = (i * 113 + 67) % 256;
        int g = (i * 179 + 89) % 256;
        int b = (i * 241 + 97) % 256;
        m_colors[i] = cv::Scalar(b, g, r);
    }
}

void Renderer::draw(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::Scalar color = (det.class_id >= 0 && det.class_id < (int)m_colors.size())
            ? m_colors[det.class_id]
            : cv::Scalar(0, 255, 0);

        cv::rectangle(frame, det.bbox, color, m_box_thickness);

        if (m_draw_labels) {
            char label[256];
            snprintf(label, sizeof(label), "%s %.2f",
                     det.class_name.c_str(), det.confidence);

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, m_font_scale, 1, &baseline);

            cv::Rect label_rect(
                det.bbox.x,
                det.bbox.y - text_size.height - 4,
                text_size.width,
                text_size.height + 2);

            cv::rectangle(frame, label_rect, color, -1);

            cv::putText(frame, label,
                        cv::Point(det.bbox.x, det.bbox.y - 4),
                        cv::FONT_HERSHEY_SIMPLEX,
                        m_font_scale,
                        cv::Scalar(255, 255, 255),
                        1, cv::LINE_AA);
        }
    }
}
