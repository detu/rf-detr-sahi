#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

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

struct Options {
    std::string model_path;
    std::string image_path;
    std::string output_path = "sahi_output.jpg";
    int workers = 0;            // parallel tile workers (0 = auto)
    int threads_per_worker = 0; // ggml threads per worker (0 = auto)
    float overlap = 0.20f;
    float conf = 0.45f;
    float nms_iou = 0.50f;
};

static int hardware_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_num_procs());
#else
    return std::max(1u, std::thread::hardware_concurrency());
#endif
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <model.gguf> <image.jpg> [options]\n"
        << "  -o, --output <path>    output image (default: sahi_output.jpg)\n"
        << "  -w, --workers <n>      tiles processed in parallel, one RF-DETR context each\n"
        << "                         (default: 2, capped by the tile count)\n"
        << "  -t, --threads <n>      ggml CPU threads per worker (default: cores / workers)\n"
        << "      --overlap <r>      tile overlap ratio (default: 0.20)\n"
        << "      --conf <c>         confidence threshold (default: 0.45)\n"
        << "      --iou <i>          NMS IoU threshold (default: 0.50)\n";
}

static bool parse_args(int argc, char** argv, Options& opt) {
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        const char* v = nullptr;
        if (a == "-h" || a == "--help") return false;
        else if (a == "-o" || a == "--output")  { if (!(v = next("--output")))  return false; opt.output_path = v; }
        else if (a == "-w" || a == "--workers") { if (!(v = next("--workers"))) return false; opt.workers = std::atoi(v); }
        else if (a == "-t" || a == "--threads") { if (!(v = next("--threads"))) return false; opt.threads_per_worker = std::atoi(v); }
        else if (a == "--overlap")              { if (!(v = next("--overlap"))) return false; opt.overlap = std::strtof(v, nullptr); }
        else if (a == "--conf")                 { if (!(v = next("--conf")))    return false; opt.conf = std::strtof(v, nullptr); }
        else if (a == "--iou")                  { if (!(v = next("--iou")))     return false; opt.nms_iou = std::strtof(v, nullptr); }
        else if (!a.empty() && a[0] == '-')     { std::cerr << "Unknown option: " << a << "\n"; return false; }
        else positional.push_back(a);
    }
    if (positional.size() < 2) return false;
    opt.model_path = positional[0];
    opt.image_path = positional[1];
    if (positional.size() > 2) opt.output_path = positional[2];  // backwards compatible
    return true;
}

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

// 2. Run RF-DETR on one tile and return detections in full-image coordinates.
//    Uses only its own context, so it is safe to call concurrently with
//    different contexts.
static bool detect_tile(rfdetr_context* ctx,
                        const rfdetr_detect_params& dparams,
                        const cv::Mat& rgb_image,
                        const SliceWindow& slice,
                        std::vector<Detection>& out) {
    // .clone() gives a contiguous HWC buffer for the raw byte pointer
    cv::Mat tile_rgb = rgb_image(cv::Rect(slice.x, slice.y, slice.width, slice.height)).clone();

    rfdetr_status st = RFDETR_OK;
    rfdetr_image* tile = rfdetr_image_from_rgb_buffer(tile_rgb.data, tile_rgb.cols, tile_rgb.rows, &st);
    if (!tile) {
        std::fprintf(stderr, "Failed to wrap tile (%d,%d): %s\n", slice.x, slice.y, rfdetr_status_str(st));
        return false;
    }

    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    st = rfdetr_detect(ctx, tile, &dparams, &dets, &n);
    rfdetr_image_free(tile);
    if (st != RFDETR_OK) {
        std::fprintf(stderr, "Detection failed on tile (%d,%d): %s\n", slice.x, slice.y, rfdetr_status_str(st));
        return false;
    }

    out.reserve(out.size() + n);
    for (size_t i = 0; i < n; ++i) {
        const rfdetr_detection& d = dets[i];
        Detection g;
        // Boxes are x1,y1,x2,y2 in tile pixels; translate to full-image coordinates
        g.box = cv::Rect2f(d.x1 + static_cast<float>(slice.x),
                           d.y1 + static_cast<float>(slice.y),
                           d.x2 - d.x1,
                           d.y2 - d.y1);
        g.confidence = d.score;
        g.class_id = static_cast<int>(d.class_id);
        g.class_name = d.class_name ? d.class_name : "";  // copy: pointer is owned by ctx
        out.push_back(std::move(g));
    }
    rfdetr_detections_free(dets, n);
    return true;
}

// 3. Class-aware Non-Maximum Suppression (NMS) across merged SAHI tiles.
//    Each class is independent, so classes are suppressed in parallel.
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

    // Flatten the map so OpenMP can index it; std::map keeps classes sorted,
    // which keeps the output order deterministic.
    std::vector<const std::vector<size_t>*> groups;
    groups.reserve(class_indices.size());
    for (const auto& kv : class_indices) groups.push_back(&kv.second);

    std::vector<std::vector<Detection>> kept(groups.size());

    #pragma omp parallel for schedule(dynamic, 1) if (groups.size() > 1)
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        const auto& indices = *groups[g];
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        boxes.reserve(indices.size());
        scores.reserve(indices.size());

        for (size_t idx : indices) {
            const auto& b = raw_dets[idx].box;
            boxes.emplace_back(static_cast<int>(std::round(b.x)),
                               static_cast<int>(std::round(b.y)),
                               static_cast<int>(std::round(b.width)),
                               static_cast<int>(std::round(b.height)));
            scores.push_back(raw_dets[idx].confidence);
        }

        std::vector<int> keep_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_iou_threshold, keep_indices);

        kept[g].reserve(keep_indices.size());
        for (int k : keep_indices) kept[g].push_back(raw_dets[indices[k]]);
    }

    for (auto& v : kept) {
        final_dets.insert(final_dets.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }
    return final_dets;
}

// 4. Draw boxes with "<class name> <score>" labels on a green background
static void draw_detections(cv::Mat& bgr_image, const std::vector<Detection>& dets) {
    const cv::Scalar GREEN(0, 255, 0);  // BGR
    const cv::Scalar BLACK(0, 0, 0);
    const int FONT = cv::FONT_HERSHEY_SIMPLEX;
    const double FONT_SCALE = 0.5;
    const int FONT_THICKNESS = 1;
    const int PAD = 3;

    for (const auto& det : dets) {
        cv::Rect box(static_cast<int>(std::round(det.box.x)), static_cast<int>(std::round(det.box.y)),
                     static_cast<int>(std::round(det.box.width)), static_cast<int>(std::round(det.box.height)));
        cv::rectangle(bgr_image, box, GREEN, 2);

        // Label: "<class name> <score>", e.g. "person 0.87"
        std::string name = det.class_name.empty() ? ("id " + std::to_string(det.class_id)) : det.class_name;
        char score[16];
        std::snprintf(score, sizeof(score), "%.2f", det.confidence);
        std::string label = name + " " + score;

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS, &baseline);
        int bg_w = ts.width + 2 * PAD;
        int bg_h = ts.height + baseline + 2 * PAD;

        // Place the label above the box; if there's no room, put it just inside the top edge.
        int bg_x = std::clamp(box.x, 0, std::max(0, bgr_image.cols - bg_w));
        int bg_y = box.y - bg_h;
        if (bg_y < 0) bg_y = std::min(std::max(box.y, 0), std::max(0, bgr_image.rows - bg_h));

        cv::rectangle(bgr_image, cv::Rect(bg_x, bg_y, bg_w, bg_h), GREEN, cv::FILLED);
        cv::putText(bgr_image, label, cv::Point(bg_x + PAD, bg_y + PAD + ts.height),
                    FONT, FONT_SCALE, BLACK, FONT_THICKNESS, cv::LINE_AA);
    }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    const uint32_t TOP_K = 300;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Load high-resolution image; RF-DETR expects RGB (HWC, 8-bit)
    cv::Mat bgr_image = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
        std::cerr << "Failed to load image: " << opt.image_path << "\n";
        return 1;
    }
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // ---- Context 0: read the model's input size so we know the tile count ----
    // The GPU backend is chosen at build time: with RFDETR_GGML_VULKAN=ON
    // (set in CMakeLists.txt) each context uses Vulkan device 0 and falls back
    // to CPU if no Vulkan device is found.
    const int hw = hardware_threads();
    int workers = opt.workers > 0 ? opt.workers : 2;
    int threads_per_worker = opt.threads_per_worker > 0 ? opt.threads_per_worker
                                                        : std::max(1, hw / workers);

    std::vector<rfdetr_context*> contexts;
    auto free_contexts = [&]() { for (auto* c : contexts) rfdetr_free(c); contexts.clear(); };

    auto make_context = [&](rfdetr_status* st) {
        rfdetr_params params{};
        params.model_path = opt.model_path.c_str();
        params.n_threads = threads_per_worker;
        return rfdetr_init(&params, st);
    };

    rfdetr_status status = RFDETR_OK;
    rfdetr_context* first = make_context(&status);
    if (!first) {
        std::cerr << "Failed to initialize RF-DETR: " << rfdetr_status_str(status) << "\n";
        return 1;
    }
    contexts.push_back(first);

    // Slice size = the model's native input resolution (384 Nano, 512 Small, 576 Medium)
    int slice_size = static_cast<int>(rfdetr_context_image_size(first));
    if (slice_size <= 0) slice_size = 576;

    auto slices = generate_slices(rgb_image.cols, rgb_image.rows, slice_size, slice_size, opt.overlap);

    // No point in more workers than tiles
    workers = std::max(1, std::min<int>(workers, static_cast<int>(slices.size())));

    // ---- Remaining contexts: one per worker (a context is not shared between threads) ----
    // Loaded serially so a failure is reported cleanly and model I/O isn't contended.
    for (int w = 1; w < workers; ++w) {
        rfdetr_context* c = make_context(&status);
        if (!c) {
            std::cerr << "Warning: could not create context " << w << " ("
                      << rfdetr_status_str(status) << "); continuing with " << w << " worker(s)\n";
            break;
        }
        contexts.push_back(c);
    }
    workers = static_cast<int>(contexts.size());

#ifdef _OPENMP
    // ggml's CPU backend parallelizes each graph with its own OpenMP region.
    // Allow one nested level so each worker's ggml call still gets its threads.
    omp_set_max_active_levels(2);
#endif

    std::cout << "Model: " << rfdetr_context_variant(first)
              << " (input " << slice_size << "x" << slice_size << ")\n"
              << "Running SAHI across " << slices.size() << " tiles with "
              << workers << " worker(s) x " << threads_per_worker << " thread(s)"
#ifndef _OPENMP
              << " [built without OpenMP: tiles run sequentially]"
#endif
              << "...\n";

    rfdetr_detect_params dparams{};
    dparams.threshold = opt.conf;
    dparams.top_k = TOP_K;
    dparams.class_filter = nullptr;
    dparams.class_filter_len = 0;

    // ---- Slice -> Infer -> Remap, tiles in parallel ----
    // Results go into per-tile slots and are concatenated in tile order, so the
    // output is identical regardless of thread scheduling.
    std::vector<std::vector<Detection>> per_tile(slices.size());
    int failed_tiles = 0;
    const auto t_infer = clock::now();

    #pragma omp parallel num_threads(workers) reduction(+ : failed_tiles)
    {
#ifdef _OPENMP
        rfdetr_context* ctx = contexts[omp_get_thread_num()];
#else
        rfdetr_context* ctx = contexts[0];
#endif
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(slices.size()); ++i) {
            if (!detect_tile(ctx, dparams, rgb_image, slices[i], per_tile[i])) ++failed_tiles;
        }
    }

    const double infer_ms = std::chrono::duration<double, std::milli>(clock::now() - t_infer).count();

    size_t total = 0;
    for (const auto& v : per_tile) total += v.size();
    std::vector<Detection> all_detections;
    all_detections.reserve(total);
    for (auto& v : per_tile) {
        all_detections.insert(all_detections.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }

    // Deduplicate overlapping detections across tile boundaries
    std::vector<Detection> merged = apply_class_aware_nms(all_detections, opt.conf, opt.nms_iou);
    std::cout << "Raw tile detections: " << all_detections.size()
              << " | Merged detections: " << merged.size();
    if (failed_tiles > 0) std::cout << " | Failed tiles: " << failed_tiles;
    std::cout << "\n";

    draw_detections(bgr_image, merged);
    cv::imwrite(opt.output_path, bgr_image);

    const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::printf("Inference: %.1f ms (%.1f ms/tile) | Total: %.1f ms\n",
                infer_ms, slices.empty() ? 0.0 : infer_ms / slices.size(), total_ms);
    std::cout << "Saved " << opt.output_path << "\n";

    free_contexts();
    return failed_tiles == static_cast<int>(slices.size()) ? 1 : 0;
}
