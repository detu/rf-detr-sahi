#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "rfdetr.h"  // C API from mudler/rf-detr.cpp (rf-detr.cpp/include/rfdetr.h)

struct Detection {
    cv::Rect2f box;  // Global (x, y, width, height)
    float confidence;
    int class_id;
    std::string class_name;
};

struct SliceWindow {
    int x;
    int y;
    int width;
    int height;
};

// 1. Generate overlapping SAHI tile coordinates
std::vector<SliceWindow> generate_slices(int img_w, int img_h, int slice_w, int slice_h, float overlap_ratio) {
    std::vector<SliceWindow> slices;
    int step_x = std::max(1, static_cast<int>(slice_w * (1.0f - overlap_ratio)));
    int step_y = std::max(1, static_cast<int>(slice_h * (1.0f - overlap_ratio)));

    for (int y = 0; y < img_h; y += step_y) {
        int h = std::min(slice_h, img_h - y);
        // Shift window back if we hit the bottom edge so slice keeps full native resolution
        int start_y = (h < slice_h && img_h >= slice_h) ? (img_h - slice_h) : y;
        h = std::min(slice_h, img_h - start_y);

        for (int x = 0; x < img_w; x += step_x) {
            int w = std::min(slice_w, img_w - x);
            // Shift window back if we hit the right edge
            int start_x = (w < slice_w && img_w >= slice_w) ? (img_w - slice_w) : x;
            w = std::min(slice_w, img_w - start_x);

            slices.push_back({start_x, start_y, w, h});

            if (start_x + w >= img_w) break;
        }
        if (start_y + h >= img_h) break;
    }
    return slices;
}

// 2. Class-aware Non-Maximum Suppression (NMS) across merged SAHI tiles
std::vector<Detection> apply_class_aware_nms(
    const std::vector<Detection>& raw_dets,
    float conf_threshold,
    float nms_iou_threshold
) {
    std::vector<Detection> final_dets;
    if (raw_dets.empty()) return final_dets;

    // Group detections by class_id so different classes don't suppress each other
    std::map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < raw_dets.size(); ++i) {
        class_indices[raw_dets[i].class_id].push_back(i);
    }

    for (const auto& [cls_id, indices] : class_indices) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        boxes.reserve(indices.size());
        scores.reserve(indices.size());

        for (size_t idx : indices) {
            const auto& b = raw_dets[idx].box;
            boxes.emplace_back(
                static_cast<int>(std::round(b.x)),
                static_cast<int>(std::round(b.y)),
                static_cast<int>(std::round(b.width)),
                static_cast<int>(std::round(b.height))
            );
            scores.push_back(raw_dets[idx].confidence);
        }

        std::vector<int> keep_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_iou_threshold, keep_indices);

        for (int k : keep_indices) {
            final_dets.push_back(raw_dets[indices[k]]);
        }
    }
    return final_dets;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <high_res_image.jpg> [output.jpg]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = (argc > 3) ? argv[3] : "sahi_output.jpg";

    // Hyperparameters
    const float OVERLAP_RATIO = 0.20f;
    const float CONF_THRESH = 0.45f;
    const float NMS_IOU_THRESH = 0.50f;
    const uint32_t TOP_K = 300;

    // 1. Load high-resolution image
    cv::Mat bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    // RF-DETR expects RGB pixel layout (HWC, 8-bit)
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // 2. Initialize RF-DETR context once.
    // The GPU backend is chosen at build time: with RFDETR_GGML_VULKAN=ON
    // (set in CMakeLists.txt) rf-detr.cpp uses Vulkan device 0 and falls back
    // to CPU if no Vulkan device is found.
    rfdetr_params params{};
    params.model_path = model_path.c_str();
    params.n_threads = 4;

    rfdetr_status status = RFDETR_OK;
    rfdetr_context* ctx = rfdetr_init(&params, &status);
    if (!ctx) {
        std::cerr << "Failed to initialize RF-DETR: " << rfdetr_status_str(status) << "\n";
        return 1;
    }

    // Slice size = the model's native input resolution (e.g. 384 Nano,
    // 512 Small, 576 Medium), read from the GGUF.
    int slice_size = static_cast<int>(rfdetr_context_image_size(ctx));
    if (slice_size <= 0) slice_size = 576;
    std::cout << "Model: " << rfdetr_context_variant(ctx)
              << " (input " << slice_size << "x" << slice_size << ")\n";

    rfdetr_detect_params dparams{};
    dparams.threshold = CONF_THRESH;
    dparams.top_k = TOP_K;
    dparams.class_filter = nullptr;
    dparams.class_filter_len = 0;

    // 3. Generate SAHI tiles
    auto slices = generate_slices(rgb_image.cols, rgb_image.rows, slice_size, slice_size, OVERLAP_RATIO);
    std::cout << "Running SAHI across " << slices.size() << " tiles ("
              << slice_size << "x" << slice_size << ")...\n";

    std::vector<Detection> all_detections;

    // 4. Slice -> Infer -> Remap coordinates
    for (const auto& slice : slices) {
        cv::Rect roi(slice.x, slice.y, slice.width, slice.height);
        // .clone() ensures a contiguous buffer for the raw byte pointer
        cv::Mat tile_rgb = rgb_image(roi).clone();

        rfdetr_status st = RFDETR_OK;
        rfdetr_image* tile = rfdetr_image_from_rgb_buffer(tile_rgb.data, tile_rgb.cols, tile_rgb.rows, &st);
        if (!tile) {
            std::cerr << "Failed to wrap tile: " << rfdetr_status_str(st) << "\n";
            continue;
        }

        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        st = rfdetr_detect(ctx, tile, &dparams, &dets, &n);
        rfdetr_image_free(tile);
        if (st != RFDETR_OK) {
            std::cerr << "Detection failed on tile (" << slice.x << "," << slice.y << "): "
                      << rfdetr_status_str(st) << "\n";
            continue;
        }

        for (size_t i = 0; i < n; ++i) {
            const rfdetr_detection& d = dets[i];
            Detection global_det;
            // Boxes are x1,y1,x2,y2 in tile pixels; translate to full-image coordinates
            global_det.box = cv::Rect2f(
                d.x1 + static_cast<float>(slice.x),
                d.y1 + static_cast<float>(slice.y),
                d.x2 - d.x1,
                d.y2 - d.y1
            );
            global_det.confidence = d.score;
            global_det.class_id = static_cast<int>(d.class_id);
            global_det.class_name = d.class_name ? d.class_name : "";
            all_detections.push_back(global_det);
        }

        rfdetr_detections_free(dets, n);
    }

    // 5. Deduplicate overlapping detections across tile boundaries
    std::vector<Detection> merged = apply_class_aware_nms(all_detections, CONF_THRESH, NMS_IOU_THRESH);
    std::cout << "Raw tile detections: " << all_detections.size()
              << " | Merged detections: " << merged.size() << "\n";

    // 6. Draw final detections and save output
    for (const auto& det : merged) {
        cv::rectangle(bgr_image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string name = det.class_name.empty() ? ("ID:" + std::to_string(det.class_id)) : det.class_name;
        std::string label = name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(bgr_image, label, cv::Point(static_cast<int>(det.box.x), static_cast<int>(std::max(0.0f, det.box.y - 5.0f))),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite(output_path, bgr_image);
    std::cout << "Saved " << output_path << "\n";

    rfdetr_free(ctx);
    return 0;
}
