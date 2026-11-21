// RF-DETR tiled inference (rf-detr.cpp / ggml).
//
// The image is cut into tiles of the model's input size on a grid with 50%
// overlap (stride = half a tile by default); the last tile in each row/column
// is clamped to the image border. Every tile runs through RF-DETR at native
// resolution. Detections touching an interior tile edge are dropped, because
// the neighbouring tile sees those objects whole; the rest are mapped back to
// the image and merged with one class-aware NMS.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
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

struct Tile {
    int x;
    int y;
    int width;
    int height;
};

namespace fs = std::filesystem;

struct Options {
    std::string model_path;
    std::vector<std::string> inputs;  // image files and/or directories
    std::string output;               // -o: output file (single image) or directory
    std::string csv_path;             // --csv: optional detections CSV
    std::string names_path;           // --names: class names, one per line (line 1 = class 0)
    bool recursive = false;
    int workers = 0;                  // parallel tile workers (0 = auto)
    int threads_per_worker = 0;       // ggml threads per worker (0 = auto)
    int stride = 0;                   // tile stride (0 = half a tile, i.e. 50% overlap)
    int margin = 8;                   // drop detections within this many px of an interior tile edge
    float pre_scale = 1.0f;           // downscale the image before tiling (fewer tiles)
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
        << "Usage: " << prog << " <model.gguf> <image|dir> [<image|dir> ...] [options]\n"
        << "\n"
        << "Loads the model once and runs tiled detection on every image given. Directories\n"
        << "are scanned for .jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp files (sorted by name).\n"
        << "\n"
        << "  -o, --output <path>    single image: output file (default: output.jpg)\n"
        << "                         several images: output directory (default: output/)\n"
        << "                         files are written as <name>_det.<ext>\n"
        << "      --csv <path>       also write all detections to a CSV file\n"
        << "  -n, --names <file>     class names, one per line: line 1 = class 0, line 2 = class 1, ...\n"
        << "                         (default: the names stored in the model)\n"
        << "  -r, --recursive        scan directories recursively\n"
        << "  -w, --workers <n>      tiles processed in parallel, one RF-DETR context each (default: 2)\n"
        << "  -t, --threads <n>      ggml CPU threads per worker (default: cores / workers)\n"
        << "      --stride <n>       tile stride in pixels (default: half a tile, i.e. 50% overlap)\n"
        << "      --margin <n>       drop detections within n px of an interior tile edge (default: 8)\n"
        << "      --pre-scale <f>    downscale the image by f (0 < f <= 1) before tiling (default: 1)\n"
        << "      --conf <c>         confidence threshold (default: 0.45)\n"
        << "      --iou <i>          NMS IoU threshold for merging the tiles (default: 0.50)\n"
        << "\n"
        << "Tiles are the model's input size (384 Nano, 512 Small, 576 Medium).\n";
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
        else if (a == "-o" || a == "--output")    { if (!(v = next("--output")))  return false; opt.output = v; }
        else if (a == "--csv")                    { if (!(v = next("--csv")))     return false; opt.csv_path = v; }
        else if (a == "-n" || a == "--names")     { if (!(v = next("--names")))   return false; opt.names_path = v; }
        else if (a == "-r" || a == "--recursive") { opt.recursive = true; }
        else if (a == "-w" || a == "--workers")   { if (!(v = next("--workers"))) return false; opt.workers = std::atoi(v); }
        else if (a == "-t" || a == "--threads")   { if (!(v = next("--threads"))) return false; opt.threads_per_worker = std::atoi(v); }
        else if (a == "--stride")                 { if (!(v = next("--stride")))  return false; opt.stride = std::atoi(v); }
        else if (a == "--margin")                 { if (!(v = next("--margin")))  return false; opt.margin = std::atoi(v); }
        else if (a == "--pre-scale")              { if (!(v = next("--pre-scale"))) return false; opt.pre_scale = std::strtof(v, nullptr); }
        else if (a == "--conf")                   { if (!(v = next("--conf")))    return false; opt.conf = std::strtof(v, nullptr); }
        else if (a == "--iou")                    { if (!(v = next("--iou")))     return false; opt.nms_iou = std::strtof(v, nullptr); }
        else if (!a.empty() && a[0] == '-')       { std::cerr << "Unknown option: " << a << "\n"; return false; }
        else positional.push_back(a);
    }
    if (opt.stride < 0) { std::cerr << "--stride must be positive\n"; return false; }
    if (opt.margin < 0) { std::cerr << "--margin must be 0 or more\n"; return false; }
    if (!(opt.pre_scale > 0.0f && opt.pre_scale <= 1.0f)) { std::cerr << "--pre-scale must be in (0, 1]\n"; return false; }
    if (positional.size() < 2) return false;
    opt.model_path = positional[0];
    opt.inputs.assign(positional.begin() + 1, positional.end());
    return true;
}

static bool is_image_file(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    static const char* exts[] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    for (const char* e : exts) if (ext == e) return true;
    return false;
}

// Expand the command-line inputs into a list of image files.
static std::vector<fs::path> collect_images(const std::vector<std::string>& inputs, bool recursive) {
    std::vector<fs::path> files;
    for (const auto& in : inputs) {
        std::error_code ec;
        fs::path p(in);
        if (fs::is_directory(p, ec)) {
            std::vector<fs::path> found;
            auto add = [&](const fs::directory_entry& e) {
                if (e.is_regular_file(ec) && is_image_file(e.path())) found.push_back(e.path());
            };
            if (recursive) {
                for (const auto& e : fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec)) add(e);
            } else {
                for (const auto& e : fs::directory_iterator(p, ec)) add(e);
            }
            std::sort(found.begin(), found.end());
            if (found.empty()) std::cerr << "Warning: no images found in " << p << "\n";
            files.insert(files.end(), found.begin(), found.end());
        } else if (fs::exists(p, ec)) {
            files.push_back(p);
        } else {
            std::cerr << "Warning: input not found, skipping: " << in << "\n";
        }
    }
    return files;
}

// Read class names: line N (1-based) is the name of class N-1. Surrounding
// whitespace and Windows line endings are stripped. Empty lines are kept, so
// line numbers stay aligned with class ids; an empty name falls back to the
// model's name for that class. A UTF-8 byte-order mark at the start is ignored.
static bool load_class_names(const std::string& path, std::vector<std::string>& names) {
    std::ifstream f(path);
    if (!f) return false;
    names.clear();
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (first && line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        first = false;
        const auto b = line.find_first_not_of(" \t\r\n");
        const auto e = line.find_last_not_of(" \t\r\n");
        names.push_back(b == std::string::npos ? std::string() : line.substr(b, e - b + 1));
    }
    // Drop trailing empty lines (e.g. a final newline at the end of the file)
    while (!names.empty() && names.back().empty()) names.pop_back();
    return true;
}

static std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + "\"";
}

// 1. Tile grid

// Number of tile positions along one axis so that the last tile ends exactly
// at the image border, without duplicate tiles.
static int tile_count(int length, int size, int stride) {
    if (length <= size) return 1;
    return (length - size + stride - 1) / stride + 1;
}

// Tile grid over an image of at least size x size pixels: a row/column every
// `stride` pixels, the last one clamped to the border.
static std::vector<Tile> tile_grid(int img_w, int img_h, int size, int stride) {
    std::vector<Tile> tiles;
    const int nx = tile_count(img_w, size, stride);
    const int ny = tile_count(img_h, size, stride);
    tiles.reserve(static_cast<size_t>(nx) * ny);
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            const int x = std::max(0, std::min(c * stride, img_w - size));
            const int y = std::max(0, std::min(r * stride, img_h - size));
            tiles.push_back({x, y, std::min(size, img_w), std::min(size, img_h)});
        }
    }
    return tiles;
}

// The image the tiles are cut from: optionally downscaled (fewer tiles), and
// padded bottom/right with gray to at least one tile, so small images still
// give a full, square tile (RF-DETR resizes its input to a square without
// keeping the aspect ratio). Padding at bottom/right keeps coordinates
// unchanged. `inv_scale` maps work-image coordinates back to the original.
static cv::Mat make_work_image(const cv::Mat& rgb, int size, float pre_scale, float& inv_scale) {
    cv::Mat work = rgb;
    inv_scale = 1.0f;
    if (pre_scale > 0.0f && pre_scale < 1.0f) {
        cv::resize(rgb, work, cv::Size(), pre_scale, pre_scale, cv::INTER_AREA);
        inv_scale = static_cast<float>(rgb.cols) / work.cols;  // exact for the rounded size
    }
    if (work.cols < size || work.rows < size) {
        cv::Mat padded;
        cv::copyMakeBorder(work, padded, 0, std::max(0, size - work.rows), 0, std::max(0, size - work.cols),
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        work = padded;
    }
    return work;
}

// 2. Run RF-DETR on one tile and return detections in work-image coordinates.
//    Uses only its own context, so it is safe to call concurrently with
//    different contexts.
static bool detect_tile(rfdetr_context* ctx,
                        const rfdetr_detect_params& dparams,
                        const cv::Mat& rgb_image,
                        const Tile& tile,
                        std::vector<Detection>& out) {
    // .clone() gives a contiguous HWC buffer for the raw byte pointer
    cv::Mat tile_rgb = rgb_image(cv::Rect(tile.x, tile.y, tile.width, tile.height)).clone();

    rfdetr_status st = RFDETR_OK;
    rfdetr_image* img = rfdetr_image_from_rgb_buffer(tile_rgb.data, tile_rgb.cols, tile_rgb.rows, &st);
    if (!img) {
        std::fprintf(stderr, "Failed to wrap tile (%d,%d): %s\n", tile.x, tile.y, rfdetr_status_str(st));
        return false;
    }

    rfdetr_detection* dets = nullptr;
    size_t n = 0;
    st = rfdetr_detect(ctx, img, &dparams, &dets, &n);
    rfdetr_image_free(img);
    if (st != RFDETR_OK) {
        std::fprintf(stderr, "Detection failed on tile (%d,%d): %s\n", tile.x, tile.y, rfdetr_status_str(st));
        return false;
    }

    out.reserve(out.size() + n);
    for (size_t i = 0; i < n; ++i) {
        const rfdetr_detection& d = dets[i];
        Detection g;
        // Boxes are x1,y1,x2,y2 in tile pixels; translate to full-image coordinates
        g.box = cv::Rect2f(d.x1 + static_cast<float>(tile.x),
                           d.y1 + static_cast<float>(tile.y),
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

// 3. Class-aware Non-Maximum Suppression (NMS) across the merged tiles.
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
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        boxes.reserve(indices.size());
        scores.reserve(indices.size());

        for (size_t idx : indices) {
            const auto& b = raw_dets[idx].box;
            boxes.emplace_back(b.x, b.y, b.width, b.height);
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

// Detect on all tiles of one image and merge. Results go into per-tile slots
// and are concatenated in tile order, so the output is identical regardless
// of thread scheduling.
static std::vector<Detection> run_tiled(const std::vector<rfdetr_context*>& contexts,
                                        const rfdetr_detect_params& dparams,
                                        const cv::Mat& rgb_image,
                                        int tile_size,
                                        const Options& opt,
                                        size_t& n_tiles, size_t& n_raw, int& failed_tiles) {
    float inv_scale = 1.0f;
    const cv::Mat work = make_work_image(rgb_image, tile_size, opt.pre_scale, inv_scale);
    const int stride = opt.stride > 0 ? opt.stride : std::max(1, tile_size / 2);
    const std::vector<Tile> tiles = tile_grid(work.cols, work.rows, tile_size, stride);
    n_tiles = tiles.size();

    // No point in more workers than tiles
    const int workers = std::max(1, std::min<int>(static_cast<int>(contexts.size()), static_cast<int>(tiles.size())));
    const float margin = static_cast<float>(opt.margin);

    std::vector<std::vector<Detection>> per_tile(tiles.size());
    int failed = 0;
    (void)workers;  // only used by the OpenMP pragma

    #pragma omp parallel num_threads(workers) reduction(+ : failed)
    {
#ifdef _OPENMP
        rfdetr_context* ctx = contexts[omp_get_thread_num()];
#else
        rfdetr_context* ctx = contexts[0];
#endif
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(tiles.size()); ++i) {
            const Tile& t = tiles[i];
            std::vector<Detection> dets;
            if (!detect_tile(ctx, dparams, work, t, dets)) { ++failed; continue; }

            // Drop objects touching an *interior* tile edge: with 50% overlap the
            // neighbouring tile sees them whole. Objects on a real image edge are kept.
            const bool left_interior = t.x > 0;
            const bool top_interior = t.y > 0;
            const bool right_interior = t.x + t.width < work.cols;
            const bool bottom_interior = t.y + t.height < work.rows;
            auto& out = per_tile[i];
            out.reserve(dets.size());
            for (auto& d : dets) {
                const float x0 = d.box.x - t.x, y0 = d.box.y - t.y;  // tile-local
                const float x1 = x0 + d.box.width, y1 = y0 + d.box.height;
                if (left_interior && x0 <= margin) continue;
                if (top_interior && y0 <= margin) continue;
                if (right_interior && x1 >= t.width - margin) continue;
                if (bottom_interior && y1 >= t.height - margin) continue;
                out.push_back(std::move(d));
            }
        }
    }
    failed_tiles = failed;

    // Back to original-image coordinates (undo the pre-scale), clipped to the
    // real image, i.e. without the gray padding.
    std::vector<Detection> all;
    const float max_x = static_cast<float>(rgb_image.cols), max_y = static_cast<float>(rgb_image.rows);
    for (auto& v : per_tile) {
        for (auto& d : v) {
            const float x0 = std::clamp(d.box.x * inv_scale, 0.0f, max_x);
            const float y0 = std::clamp(d.box.y * inv_scale, 0.0f, max_y);
            const float x1 = std::clamp((d.box.x + d.box.width) * inv_scale, 0.0f, max_x);
            const float y1 = std::clamp((d.box.y + d.box.height) * inv_scale, 0.0f, max_y);
            if (x1 - x0 < 1.0f || y1 - y0 < 1.0f) continue;
            d.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
            all.push_back(std::move(d));
        }
    }
    n_raw = all.size();

    // Duplicates of objects seen whole by two overlapping tiles
    return apply_class_aware_nms(all, opt.conf, opt.nms_iou);
}

struct LoadedImage {
    cv::Mat bgr;
    cv::Mat rgb;
};

static LoadedImage load_image(const fs::path& path) {
    LoadedImage li;
    li.bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (!li.bgr.empty()) cv::cvtColor(li.bgr, li.rgb, cv::COLOR_BGR2RGB);  // RF-DETR expects RGB (HWC, 8-bit)
    return li;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::vector<fs::path> images = collect_images(opt.inputs, opt.recursive);
    if (images.empty()) {
        std::cerr << "No input images.\n";
        return 1;
    }

    // Optional class-name file overrides the names stored in the model.
    // Read before loading the model so a bad path fails immediately.
    std::vector<std::string> class_names;
    if (!opt.names_path.empty() && !load_class_names(opt.names_path, class_names)) {
        std::cerr << "Cannot read class names file: " << opt.names_path << "\n";
        return 1;
    }

    // Output naming: one image -> -o is a file; several -> -o is a directory.
    const bool single = images.size() == 1;
    fs::path out_file, out_dir;
    if (single) {
        out_file = opt.output.empty() ? fs::path("output.jpg") : fs::path(opt.output);
        if (fs::is_directory(out_file)) out_file /= images[0].stem().string() + "_det" + images[0].extension().string();
    } else {
        out_dir = opt.output.empty() ? fs::path("output") : fs::path(opt.output);
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            std::cerr << "Cannot create output directory " << out_dir << ": " << ec.message() << "\n";
            return 1;
        }
    }
    // Several inputs can share a file name (e.g. in different subdirectories with -r):
    // add _2, _3, ... so outputs never overwrite each other.
    std::map<std::string, int> used_names;
    auto output_for = [&](const fs::path& in) -> fs::path {
        if (single) return out_file;
        std::string base = in.stem().string() + "_det";
        std::string ext = in.extension().string();
        int& n = used_names[base + ext];
        std::string name = (++n == 1) ? base + ext : base + "_" + std::to_string(n) + ext;
        return out_dir / name;
    };

    // ---- Create the RF-DETR contexts once; they are reused for every image ----
    // The GPU backend is chosen at build time (SAHI_BACKEND in CMakeLists.txt);
    // each context uses GPU device 0 and falls back to CPU if none is found.
    const int hw = hardware_threads();
    int workers = opt.workers > 0 ? opt.workers : 2;
    const int threads_per_worker = opt.threads_per_worker > 0 ? opt.threads_per_worker
                                                              : std::max(1, hw / workers);

    auto make_context = [&](rfdetr_status* st) {
        rfdetr_params params{};
        params.model_path = opt.model_path.c_str();
        params.n_threads = threads_per_worker;
        return rfdetr_init(&params, st);
    };

    std::vector<rfdetr_context*> contexts;
    rfdetr_status status = RFDETR_OK;
    // Loaded serially so a failure is reported cleanly and model I/O isn't contended.
    for (int w = 0; w < workers; ++w) {
        rfdetr_context* c = make_context(&status);
        if (!c) {
            if (w == 0) {
                std::cerr << "Failed to initialize RF-DETR: " << rfdetr_status_str(status) << "\n";
                return 1;
            }
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

    // Tile size = the model's native input resolution (384 Nano, 512 Small, 576 Medium)
    int tile_size = static_cast<int>(rfdetr_context_image_size(contexts[0]));
    if (tile_size <= 0) tile_size = 576;
    const int stride = opt.stride > 0 ? opt.stride : std::max(1, tile_size / 2);

    rfdetr_detect_params dparams{};
    dparams.threshold = opt.conf;
    dparams.top_k = 300;
    dparams.class_filter = nullptr;
    dparams.class_filter_len = 0;

    const double init_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::cout << "Model: " << rfdetr_context_variant(contexts[0])
              << " (input " << tile_size << "x" << tile_size << ") | tiles " << tile_size << ", stride " << stride
              << ", edge margin " << opt.margin
              << (opt.pre_scale < 1.0f ? ", pre-scale " + std::to_string(opt.pre_scale).substr(0, 4) : std::string())
              << " | "
              << workers << " worker(s) x " << threads_per_worker << " thread(s)"
#ifndef _OPENMP
              << " [built without OpenMP: tiles run sequentially]"
#endif
              << " | " << images.size() << " image(s) | init " << static_cast<int>(init_ms) << " ms\n";

    if (!opt.names_path.empty()) {
        std::cout << "Class names: " << class_names.size() << " from " << opt.names_path;
        const uint32_t model_classes = rfdetr_context_num_classes(contexts[0]);
        if (model_classes > 0 && class_names.size() != model_classes) {
            std::cout << " (model has " << model_classes << " classes; ids without a name keep the model's name)";
        }
        std::cout << "\n";
    }
    size_t unnamed_ids = 0;  // detections whose class id has no line in the names file

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        csv.open(opt.csv_path);
        if (!csv) {
            std::cerr << "Cannot open CSV file: " << opt.csv_path << "\n";
            for (auto* c : contexts) rfdetr_free(c);
            return 1;
        }
        csv << "image,class_id,class_name,score,x1,y1,x2,y2\n";
    }

    // ---- Process images; decode the next one in the background meanwhile ----
    size_t ok_images = 0, failed_images = 0, total_dets = 0;
    double total_infer_ms = 0.0;
    std::future<LoadedImage> next = std::async(std::launch::async, load_image, images[0]);

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const fs::path& path = images[idx];
        LoadedImage img = next.get();
        if (idx + 1 < images.size()) next = std::async(std::launch::async, load_image, images[idx + 1]);

        std::cout << "[" << (idx + 1) << "/" << images.size() << "] " << path.string() << ": ";
        if (img.bgr.empty()) {
            std::cout << "failed to load, skipped\n";
            ++failed_images;
            continue;
        }

        size_t n_tiles = 0, n_raw = 0;
        int failed_tiles = 0;
        const auto ti = clock::now();
        std::vector<Detection> merged = run_tiled(contexts, dparams, img.rgb, tile_size, opt, n_tiles, n_raw, failed_tiles);
        const double infer_ms = std::chrono::duration<double, std::milli>(clock::now() - ti).count();

        if (!class_names.empty()) {
            for (auto& d : merged) {
                if (d.class_id >= 0 && static_cast<size_t>(d.class_id) < class_names.size() &&
                    !class_names[d.class_id].empty()) {
                    d.class_name = class_names[d.class_id];
                } else {
                    ++unnamed_ids;
                }
            }
        }

        draw_detections(img.bgr, merged);
        const fs::path out = output_for(path);
        bool saved = false;
        try {
            saved = cv::imwrite(out.string(), img.bgr);
        } catch (const cv::Exception& e) {
            std::cerr << "imwrite error: " << e.what() << "\n";
        }

        if (csv.is_open()) {
            for (const auto& d : merged) {
                csv << csv_escape(path.string()) << ',' << d.class_id << ',' << csv_escape(d.class_name) << ','
                    << d.confidence << ',' << d.box.x << ',' << d.box.y << ','
                    << (d.box.x + d.box.width) << ',' << (d.box.y + d.box.height) << '\n';
            }
        }

        std::printf("%zu tiles, %zu raw -> %zu dets, %.1f ms%s%s\n",
                    n_tiles, n_raw, merged.size(), infer_ms,
                    failed_tiles ? (" | " + std::to_string(failed_tiles) + " tile(s) failed").c_str() : "",
                    saved ? "" : " | could not save output");
        std::fflush(stdout);

        if (failed_tiles == static_cast<int>(n_tiles) || !saved) ++failed_images; else ++ok_images;
        total_dets += merged.size();
        total_infer_ms += infer_ms;
    }

    for (auto* c : contexts) rfdetr_free(c);

    const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::printf("Done: %zu ok, %zu failed, %zu detections | inference %.1f s (%.1f ms/image) | total %.1f s\n",
                ok_images, failed_images, total_dets, total_infer_ms / 1000.0,
                images.empty() ? 0.0 : total_infer_ms / images.size(), total_ms / 1000.0);
    if (unnamed_ids > 0) {
        std::cout << "Note: " << unnamed_ids << " detection(s) had a class id with no name in "
                  << opt.names_path << "; the model's name was used for those\n";
    }
    std::cout << "Output: " << (single ? out_file.string() : out_dir.string())
              << (opt.csv_path.empty() ? "" : " | CSV: " + opt.csv_path) << "\n";
    return failed_images == 0 ? 0 : 1;
}
