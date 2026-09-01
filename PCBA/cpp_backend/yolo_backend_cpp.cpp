// =============================================================================
//  Test Path B - C++17 Inference Engine
//  Project: Comparative Performance Analysis of Python and C++ Implementations
//           for High-Resolution 2D AOI in PCB Inspection
//
//  Responsibilities:
//    - Receive a 4K (3840x2160) frame from the PyQt5 frontend (zero-copy)
//    - Slice it into overlapping 1080p SAHI tiles
//    - Run YOLOv8 ONNX inference per tile via OpenCV DNN (CUDA target)
//    - Remap tile-local boxes to master-frame coordinates, global NMS
//    - Log latency / CPU / GPU / memory metrics to CSV for the perf matrix
//
//  Build (MSVC, standalone benchmark):
//    cl /std:c++17 /O2 /EHsc /DNDEBUG yolo_backend_cpp.cpp ^
//       /I"%OPENCV_DIR%\include" /link /LIBPATH:"%OPENCV_DIR%\x64\vc17\lib" ^
//       opencv_world4100.lib psapi.lib
//
//  Optional GPU utilisation via NVML (Nvidia RTX):
//    add /DUSE_NVML /I"%CUDA_PATH%\include" and link nvml.lib
//
//  Optional Python module for the Pybind11 bridge:
//    add /DBUILD_PYTHON_MODULE and build as a shared library (.pyd)
// =============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef _WIN32
  #include <windows.h>
  #include <psapi.h>
#else
  #include <unistd.h>
  #include <sys/times.h>
#endif

#ifdef USE_NVML
  #include <nvml.h>
#endif

using namespace std;
using namespace cv;
using namespace cv::dnn;

// -----------------------------------------------------------------------------
// Metric containers
// -----------------------------------------------------------------------------

// Per-tile timing breakdown. Split into stages so the final matrix can show
// *where* C++ wins or loses against Python, not just the aggregate number.
struct TileMetrics {
    int    frame_id        = 0;
    int    tile_id         = 0;
    int    tile_x          = 0;   // tile origin in master-frame coords
    int    tile_y          = 0;
    double preprocess_ms   = 0.0; // letterbox + blobFromImage
    double inference_ms    = 0.0; // net.forward()
    double postprocess_ms  = 0.0; // decode + NMS + coordinate remap
    double total_ms        = 0.0;
    int    detections      = 0;
};

struct FrameMetrics {
    int    frame_id        = 0;
    int    tiles           = 0;
    double preprocess_ms   = 0.0;
    double inference_ms    = 0.0;
    double postprocess_ms  = 0.0;
    double slicing_ms      = 0.0; // SAHI grid generation
    double merge_ms        = 0.0; // cross-tile global NMS
    double total_ms        = 0.0;
    double fps             = 0.0;
    double cpu_percent     = 0.0;
    double gpu_percent     = 0.0;
    double gpu_mem_mb      = 0.0;
    double rss_mb          = 0.0; // resident set / working set
    double peak_rss_mb     = 0.0;
    int    detections      = 0;
};

struct Detection {
    Rect  box;          // master-frame coordinates
    float confidence = 0.f;
    int   class_id   = -1;
    int   tile_id    = -1;
};

// -----------------------------------------------------------------------------
// SystemMonitor - real hardware hooks (replaces the [Pending OS Hook] stubs)
// -----------------------------------------------------------------------------
// CPU% is process-relative and normalised by core count, so a fully saturated
// 12th Gen i5 reads ~100% rather than ~1200%. This keeps the Python and C++
// numbers directly comparable on the same machine.
class SystemMonitor {
public:
    SystemMonitor() {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        num_processors_ = static_cast<int>(si.dwNumberOfProcessors);
#else
        num_processors_ = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
        if (num_processors_ <= 0) num_processors_ = 1;
        reset_cpu_baseline();

#ifdef USE_NVML
        nvml_ok_ = (nvmlInit_v2() == NVML_SUCCESS) &&
                   (nvmlDeviceGetHandleByIndex_v2(0, &nvml_device_) == NVML_SUCCESS);
        if (!nvml_ok_)
            cerr << "[monitor] NVML unavailable - GPU metrics will report -1." << endl;
#endif
    }

    ~SystemMonitor() {
#ifdef USE_NVML
        if (nvml_ok_) nvmlShutdown();
#endif
    }

    // Call once before the measured region so the next cpu_percent() call
    // reports utilisation over that window only.
    void reset_cpu_baseline() {
        last_wall_ = chrono::steady_clock::now();
        last_cpu_ns_ = process_cpu_ns();
    }

    double cpu_percent() {
        auto now = chrono::steady_clock::now();
        long long cpu_ns = process_cpu_ns();

        double wall_ns = chrono::duration<double, nano>(now - last_wall_).count();
        double busy_ns = static_cast<double>(cpu_ns - last_cpu_ns_);

        last_wall_ = now;
        last_cpu_ns_ = cpu_ns;

        if (wall_ns <= 0.0) return 0.0;
        return (busy_ns / wall_ns) * 100.0 / num_processors_;
    }

    // Working set (Windows) / RSS (Linux) in MB. This is the figure that
    // exposes the 4K buffer + SAHI tile pyramid cost.
    double rss_mb() const {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                 sizeof(pmc)))
            return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        return -1.0;
#else
        ifstream statm("/proc/self/statm");
        long size = 0, resident = 0;
        if (statm >> size >> resident)
            return static_cast<double>(resident * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
        return -1.0;
#endif
    }

    double peak_rss_mb() const {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                                 reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                 sizeof(pmc)))
            return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
        return -1.0;
#else
        return rss_mb();
#endif
    }

    double gpu_percent() const {
#ifdef USE_NVML
        if (nvml_ok_) {
            nvmlUtilization_t util{};
            if (nvmlDeviceGetUtilizationRates(nvml_device_, &util) == NVML_SUCCESS)
                return static_cast<double>(util.gpu);
        }
#endif
        return -1.0;
    }

    double gpu_mem_mb() const {
#ifdef USE_NVML
        if (nvml_ok_) {
            nvmlMemory_t mem{};
            if (nvmlDeviceGetMemoryInfo(nvml_device_, &mem) == NVML_SUCCESS)
                return static_cast<double>(mem.used) / (1024.0 * 1024.0);
        }
#endif
        return -1.0;
    }

private:
    long long process_cpu_ns() const {
#ifdef _WIN32
        FILETIME c, e, k, u;
        if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
        ULARGE_INTEGER kt, ut;
        kt.LowPart = k.dwLowDateTime;  kt.HighPart = k.dwHighDateTime;
        ut.LowPart = u.dwLowDateTime;  ut.HighPart = u.dwHighDateTime;
        // FILETIME ticks are 100 ns
        return static_cast<long long>((kt.QuadPart + ut.QuadPart) * 100ULL);
#else
        struct tms t;
        times(&t);
        long ticks = sysconf(_SC_CLK_TCK);
        if (ticks <= 0) return 0;
        return static_cast<long long>((t.tms_utime + t.tms_stime) * (1000000000LL / ticks));
#endif
    }

    int num_processors_ = 1;
    chrono::steady_clock::time_point last_wall_;
    long long last_cpu_ns_ = 0;

#ifdef USE_NVML
    bool nvml_ok_ = false;
    nvmlDevice_t nvml_device_{};
#endif
};

// -----------------------------------------------------------------------------
// BenchmarkLogger - CSV output feeding the final performance matrix
// -----------------------------------------------------------------------------
class BenchmarkLogger {
public:
    BenchmarkLogger(const string& frame_csv, const string& tile_csv) {
        frame_out_.open(frame_csv, ios::out | ios::trunc);
        tile_out_.open(tile_csv, ios::out | ios::trunc);

        frame_out_ << "frame_id,tiles,slicing_ms,preprocess_ms,inference_ms,"
                   << "postprocess_ms,merge_ms,total_ms,fps,cpu_percent,"
                   << "gpu_percent,gpu_mem_mb,rss_mb,peak_rss_mb,detections\n";
        tile_out_  << "frame_id,tile_id,tile_x,tile_y,preprocess_ms,"
                   << "inference_ms,postprocess_ms,total_ms,detections\n";
    }

    void log(const FrameMetrics& m) {
        frame_out_ << m.frame_id << ',' << m.tiles << ','
                   << fixed << setprecision(3)
                   << m.slicing_ms << ',' << m.preprocess_ms << ','
                   << m.inference_ms << ',' << m.postprocess_ms << ','
                   << m.merge_ms << ',' << m.total_ms << ','
                   << setprecision(2) << m.fps << ',' << m.cpu_percent << ','
                   << m.gpu_percent << ',' << m.gpu_mem_mb << ','
                   << m.rss_mb << ',' << m.peak_rss_mb << ','
                   << m.detections << '\n';
        frame_out_.flush();
    }

    void log(const TileMetrics& m) {
        tile_out_ << m.frame_id << ',' << m.tile_id << ',' << m.tile_x << ','
                  << m.tile_y << ',' << fixed << setprecision(3)
                  << m.preprocess_ms << ',' << m.inference_ms << ','
                  << m.postprocess_ms << ',' << m.total_ms << ','
                  << m.detections << '\n';
    }

private:
    ofstream frame_out_, tile_out_;
};

// -----------------------------------------------------------------------------
// YoloBackendCpp
// -----------------------------------------------------------------------------
class YoloBackendCpp {
public:
    struct Config {
        string model_path;
        int    input_size      = 640;   // YOLOv8 network input (square)
        bool   use_cuda        = true;
        bool   use_fp16        = true;  // DNN_TARGET_CUDA_FP16 on RTX
        float  conf_threshold  = 0.25f;
        float  nms_threshold   = 0.45f;
        // SAHI slicing parameters - must mirror the Python path exactly
        int    tile_width      = 1920;
        int    tile_height     = 1080;
        float  overlap_ratio   = 0.20f;
        int    warmup_runs     = 5;
    };

    explicit YoloBackendCpp(const Config& cfg) : cfg_(cfg) {
        net_ = readNetFromONNX(cfg_.model_path);
        if (net_.empty())
            throw runtime_error("Failed to load ONNX model: " + cfg_.model_path);

        configure_backend();
        cout << "[engine] YOLOv8 ONNX loaded: " << cfg_.model_path << endl;
        cout << "[engine] Backend: " << active_backend_ << endl;
    }

    const string& active_backend() const { return active_backend_; }

    void set_class_names(const vector<string>& names) { class_names_ = names; }

    // The first forward pass on a CUDA target pays for kernel autotuning and
    // cuDNN workspace allocation. Excluding it is what keeps the comparison
    // against Python honest - Ultralytics warms up internally too.
    void warmup() {
        Mat dummy(cfg_.tile_height, cfg_.tile_width, CV_8UC3, Scalar(114, 114, 114));
        for (int i = 0; i < cfg_.warmup_runs; ++i) {
            TileMetrics scratch;
            run_tile(dummy, Point(0, 0), 0, scratch);
        }
        cout << "[engine] Warmup complete (" << cfg_.warmup_runs << " passes)." << endl;
    }

    // ---- SAHI tile grid -----------------------------------------------------
    // MUST stay byte-for-byte equivalent to _axis_origins() in aoi_gui/engines.py.
    // If the two drift apart the paths process different pixels and the whole
    // comparison is void.
    //
    // Naive 'step forward, clamp last tile inward' gives a lopsided grid: on a
    // 3840px axis with 1920px tiles it yields 0/1536/1920, where the last pair
    // overlaps 80% and the first 20%. Compute the minimum tile count that meets
    // the requested overlap, then spread evenly - uniform overlap, always >=
    // the requested ratio.
    static vector<int> axis_origins(int total, int tile, float overlap) {
        if (tile >= total) return {0};

        const int step = max(1, static_cast<int>(tile * (1.0f - overlap)));
        const int span = total - tile;
        int n = (span + step - 1) / step + 1;      // ceil(span / step) + 1
        n = max(2, n);

        vector<int> origins;
        origins.reserve(n);
        for (int i = 0; i < n; ++i)
            origins.push_back(static_cast<int>(lround(
                static_cast<double>(i) * span / (n - 1))));
        return origins;
    }

    vector<Point> compute_slice_origins(const Size& frame_size) const {
        vector<int> xs = axis_origins(frame_size.width,  cfg_.tile_width,  cfg_.overlap_ratio);
        vector<int> ys = axis_origins(frame_size.height, cfg_.tile_height, cfg_.overlap_ratio);

        vector<Point> origins;
        origins.reserve(xs.size() * ys.size());
        for (int y : ys)
            for (int x : xs)
                origins.emplace_back(x, y);
        return origins;
    }

    // ---- Single tile --------------------------------------------------------
    vector<Detection> run_tile(const Mat& tile, const Point& origin,
                               int tile_id, TileMetrics& tm) {
        auto t0 = chrono::high_resolution_clock::now();

        // 1. Pre-processing: letterbox preserves the 16:9 tile aspect ratio.
        //    A plain resize to 640x640 would squash fine-pitch components and
        //    change the detections relative to the Python path.
        float scale = 1.f; int pad_x = 0, pad_y = 0;
        Mat padded = letterbox(tile, Size(cfg_.input_size, cfg_.input_size),
                               scale, pad_x, pad_y);
        Mat blob;
        blobFromImage(padded, blob, 1.0 / 255.0, padded.size(),
                      Scalar(), /*swapRB=*/true, /*crop=*/false);
        net_.setInput(blob);

        auto t1 = chrono::high_resolution_clock::now();

        // 2. Execution
        vector<Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        auto t2 = chrono::high_resolution_clock::now();

        // 3. Post-processing: decode, NMS, remap to master-frame coordinates
        vector<Detection> dets = decode_outputs(outputs, tile.size(),
                                                scale, pad_x, pad_y);
        for (auto& d : dets) {
            d.box.x  += origin.x;
            d.box.y  += origin.y;
            d.tile_id = tile_id;
        }

        auto t3 = chrono::high_resolution_clock::now();

        tm.tile_id        = tile_id;
        tm.tile_x         = origin.x;
        tm.tile_y         = origin.y;
        tm.preprocess_ms  = chrono::duration<double, milli>(t1 - t0).count();
        tm.inference_ms   = chrono::duration<double, milli>(t2 - t1).count();
        tm.postprocess_ms = chrono::duration<double, milli>(t3 - t2).count();
        tm.total_ms       = chrono::duration<double, milli>(t3 - t0).count();
        tm.detections     = static_cast<int>(dets.size());

        return dets;
    }

    // ---- Full 4K frame ------------------------------------------------------
    vector<Detection> run_frame(const Mat& frame_4k, int frame_id,
                                FrameMetrics& fm,
                                BenchmarkLogger* logger = nullptr) {
        if (frame_4k.empty())
            throw runtime_error("run_frame received an empty frame.");

        monitor_.reset_cpu_baseline();
        auto frame_start = chrono::high_resolution_clock::now();

        auto s0 = chrono::high_resolution_clock::now();
        vector<Point> origins = compute_slice_origins(frame_4k.size());
        auto s1 = chrono::high_resolution_clock::now();

        vector<Detection> all;
        fm = FrameMetrics{};
        fm.frame_id   = frame_id;
        fm.tiles      = static_cast<int>(origins.size());
        fm.slicing_ms = chrono::duration<double, milli>(s1 - s0).count();

        for (size_t i = 0; i < origins.size(); ++i) {
            // ROI is a view into the parent frame - no pixel copy. This is the
            // in-place memory behaviour being measured against Python's
            // per-slice array allocations.
            Rect roi(origins[i].x, origins[i].y, cfg_.tile_width, cfg_.tile_height);
            Mat tile = frame_4k(roi);

            TileMetrics tm;
            tm.frame_id = frame_id;
            vector<Detection> dets = run_tile(tile, origins[i], static_cast<int>(i), tm);

            fm.preprocess_ms  += tm.preprocess_ms;
            fm.inference_ms   += tm.inference_ms;
            fm.postprocess_ms += tm.postprocess_ms;

            all.insert(all.end(), dets.begin(), dets.end());
            if (logger) logger->log(tm);
        }

        // Cross-tile NMS: a solder bridge sitting in an overlap region is
        // detected twice, once per neighbouring tile.
        auto m0 = chrono::high_resolution_clock::now();
        vector<Detection> merged = global_nms(all);
        auto m1 = chrono::high_resolution_clock::now();

        auto frame_end = chrono::high_resolution_clock::now();

        fm.merge_ms    = chrono::duration<double, milli>(m1 - m0).count();
        fm.total_ms    = chrono::duration<double, milli>(frame_end - frame_start).count();
        fm.fps         = fm.total_ms > 0.0 ? 1000.0 / fm.total_ms : 0.0;
        fm.detections  = static_cast<int>(merged.size());
        fm.cpu_percent = monitor_.cpu_percent();
        fm.gpu_percent = monitor_.gpu_percent();
        fm.gpu_mem_mb  = monitor_.gpu_mem_mb();
        fm.rss_mb      = monitor_.rss_mb();
        fm.peak_rss_mb = monitor_.peak_rss_mb();

        if (logger) logger->log(fm);
        return merged;
    }

    void draw(Mat& frame, const vector<Detection>& dets) const {
        for (const auto& d : dets) {
            rectangle(frame, d.box, Scalar(0, 255, 0), 2);
            string label = (d.class_id >= 0 && d.class_id < (int)class_names_.size())
                         ? class_names_[d.class_id]
                         : ("cls" + to_string(d.class_id));
            ostringstream ss;
            ss << label << ' ' << fixed << setprecision(2) << d.confidence;
            putText(frame, ss.str(), Point(d.box.x, max(0, d.box.y - 6)),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
    }

private:
    void configure_backend() {
        if (cfg_.use_cuda) {
            net_.setPreferableBackend(DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cfg_.use_fp16 ? DNN_TARGET_CUDA_FP16
                                                   : DNN_TARGET_CUDA);
            // A CUDA-less OpenCV build silently falls back to CPU, which would
            // quietly corrupt the benchmark. Probe once and fail loudly.
            try {
                Mat probe = Mat::zeros(cfg_.input_size, cfg_.input_size, CV_8UC3);
                Mat blob;
                blobFromImage(probe, blob, 1.0 / 255.0, probe.size(), Scalar(), true, false);
                net_.setInput(blob);
                vector<Mat> out;
                net_.forward(out, net_.getUnconnectedOutLayersNames());
                active_backend_ = cfg_.use_fp16 ? "CUDA (FP16)" : "CUDA (FP32)";
                return;
            } catch (const cv::Exception& e) {
                cerr << "[engine] CUDA target unavailable, falling back to CPU: "
                     << e.what() << endl;
            }
        }
        net_.setPreferableBackend(DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(DNN_TARGET_CPU);
        active_backend_ = "CPU";
    }

    static Mat letterbox(const Mat& src, const Size& new_shape,
                         float& scale, int& pad_x, int& pad_y) {
        scale = min(static_cast<float>(new_shape.width)  / src.cols,
                    static_cast<float>(new_shape.height) / src.rows);
        int nw = static_cast<int>(round(src.cols * scale));
        int nh = static_cast<int>(round(src.rows * scale));

        Mat resized;
        resize(src, resized, Size(nw, nh), 0, 0, INTER_LINEAR);

        pad_x = (new_shape.width  - nw) / 2;
        pad_y = (new_shape.height - nh) / 2;

        Mat out;
        copyMakeBorder(resized, out, pad_y, new_shape.height - nh - pad_y,
                       pad_x, new_shape.width - nw - pad_x,
                       BORDER_CONSTANT, Scalar(114, 114, 114));
        return out;
    }

    // YOLOv8 exports a single output of shape [1, 4 + num_classes, num_anchors].
    // There is no objectness channel (unlike YOLOv5) - class score is the score.
    vector<Detection> decode_outputs(const vector<Mat>& outputs,
                                     const Size& tile_size,
                                     float scale, int pad_x, int pad_y) const {
        vector<Detection> result;
        if (outputs.empty()) return result;

        Mat out = outputs[0];
        if (out.dims == 3)
            out = out.reshape(1, out.size[1]);   // (4+nc) x anchors

        // Transpose to anchors x (4+nc) for row-wise iteration.
        if (out.rows < out.cols)
            transpose(out, out);

        const int num_classes = out.cols - 4;
        if (num_classes <= 0) return result;

        vector<Rect>  boxes;
        vector<float> scores;
        vector<int>   class_ids;
        boxes.reserve(64); scores.reserve(64); class_ids.reserve(64);

        for (int i = 0; i < out.rows; ++i) {
            const float* row = out.ptr<float>(i);

            Mat class_scores(1, num_classes, CV_32F, const_cast<float*>(row + 4));
            Point max_loc; double max_score = 0.0;
            minMaxLoc(class_scores, nullptr, &max_score, nullptr, &max_loc);
            if (max_score < cfg_.conf_threshold) continue;

            // cx, cy, w, h in letterboxed 640-space -> undo pad, undo scale
            float cx = (row[0] - pad_x) / scale;
            float cy = (row[1] - pad_y) / scale;
            float w  =  row[2] / scale;
            float h  =  row[3] / scale;

            int left = static_cast<int>(round(cx - w * 0.5f));
            int top  = static_cast<int>(round(cy - h * 0.5f));
            Rect box(left, top, static_cast<int>(round(w)), static_cast<int>(round(h)));
            box &= Rect(0, 0, tile_size.width, tile_size.height); // clip to tile
            if (box.width <= 0 || box.height <= 0) continue;

            boxes.push_back(box);
            scores.push_back(static_cast<float>(max_score));
            class_ids.push_back(max_loc.x);
        }

        vector<int> keep;
        NMSBoxes(boxes, scores, cfg_.conf_threshold, cfg_.nms_threshold, keep);
        result.reserve(keep.size());
        for (int idx : keep) {
            Detection d;
            d.box        = boxes[idx];
            d.confidence = scores[idx];
            d.class_id   = class_ids[idx];
            result.push_back(d);
        }
        return result;
    }

    // Class-aware NMS across the stitched master frame.
    vector<Detection> global_nms(const vector<Detection>& dets) const {
        if (dets.size() < 2) return dets;

        vector<Rect>  boxes;
        vector<float> scores;
        vector<int>   ids;
        boxes.reserve(dets.size()); scores.reserve(dets.size()); ids.reserve(dets.size());
        for (const auto& d : dets) {
            boxes.push_back(d.box);
            scores.push_back(d.confidence);
            ids.push_back(d.class_id);
        }

        vector<int> keep;
        NMSBoxesBatched(boxes, scores, ids, cfg_.conf_threshold,
                        cfg_.nms_threshold, keep);

        vector<Detection> merged;
        merged.reserve(keep.size());
        for (int idx : keep) merged.push_back(dets[idx]);
        return merged;
    }

    Config          cfg_;
    Net             net_;
    string          active_backend_ = "unset";
    vector<string>  class_names_;
    mutable SystemMonitor monitor_;
};

// -----------------------------------------------------------------------------
// Aggregate reporting
// -----------------------------------------------------------------------------
static void print_summary(const vector<FrameMetrics>& runs, const string& backend) {
    if (runs.empty()) return;

    auto mean = [&](double FrameMetrics::*field) {
        double s = 0.0;
        for (const auto& r : runs) s += r.*field;
        return s / runs.size();
    };

    vector<double> totals;
    totals.reserve(runs.size());
    for (const auto& r : runs) totals.push_back(r.total_ms);
    sort(totals.begin(), totals.end());
    double p50 = totals[totals.size() / 2];
    double p95 = totals[min(totals.size() - 1, static_cast<size_t>(totals.size() * 0.95))];

    cout << "\n================ Test Path B (C++) Summary ================\n"
         << "Backend               : " << backend << '\n'
         << "Frames                : " << runs.size() << '\n'
         << "Tiles per frame       : " << runs.front().tiles << '\n'
         << fixed << setprecision(2)
         << "Mean frame latency    : " << mean(&FrameMetrics::total_ms)   << " ms\n"
         << "  p50 / p95           : " << p50 << " / " << p95 << " ms\n"
         << "Mean effective FPS    : " << mean(&FrameMetrics::fps)        << '\n'
         << "  -> preprocess       : " << mean(&FrameMetrics::preprocess_ms)  << " ms\n"
         << "  -> inference        : " << mean(&FrameMetrics::inference_ms)   << " ms\n"
         << "  -> postprocess      : " << mean(&FrameMetrics::postprocess_ms) << " ms\n"
         << "  -> slicing / merge  : " << mean(&FrameMetrics::slicing_ms) << " / "
                                       << mean(&FrameMetrics::merge_ms)   << " ms\n"
         << "Mean CPU load         : " << mean(&FrameMetrics::cpu_percent) << " %\n"
         << "Mean GPU load         : " << mean(&FrameMetrics::gpu_percent) << " %  (-1 = NVML off)\n"
         << "Mean GPU memory       : " << mean(&FrameMetrics::gpu_mem_mb)  << " MB\n"
         << "Mean working set      : " << mean(&FrameMetrics::rss_mb)      << " MB\n"
         << "Peak working set      : " << runs.back().peak_rss_mb          << " MB\n"
         << "===========================================================\n";
}

// -----------------------------------------------------------------------------
// Standalone benchmark harness
// -----------------------------------------------------------------------------
#ifndef BUILD_PYTHON_MODULE
int main(int argc, char** argv) {
    YoloBackendCpp::Config cfg;
    cfg.model_path = (argc > 1) ? argv[1]
                                : "../datasets/weights/yolov8_pcb_4k.onnx";
    string source  = (argc > 2) ? argv[2] : "";   // image/video path, empty = synthetic
    const int frames = (argc > 3) ? atoi(argv[3]) : 30;

    try {
        YoloBackendCpp engine(cfg);
        engine.set_class_names({"missing_hole", "mouse_bite", "open_circuit",
                                "short", "spur", "spurious_copper", "tombstone"});
        engine.warmup();

        BenchmarkLogger logger("benchmark_cpp_frames.csv", "benchmark_cpp_tiles.csv");

        Mat frame;
        VideoCapture cap;
        if (!source.empty()) {
            cap.open(source);
            if (!cap.isOpened()) throw runtime_error("Cannot open source: " + source);
        } else {
            // Synthetic 4K frame so the harness runs before the camera is wired in.
            frame = Mat(2160, 3840, CV_8UC3, Scalar(30, 30, 30));
            randu(frame, Scalar(0, 0, 0), Scalar(255, 255, 255));
            cout << "[bench] No source given - using synthetic 3840x2160 frame." << endl;
        }

        vector<FrameMetrics> runs;
        runs.reserve(frames);
        cout << "[bench] Starting Test Path B over " << frames << " frames..." << endl;

        for (int i = 0; i < frames; ++i) {
            if (cap.isOpened()) {
                if (!cap.read(frame) || frame.empty()) break;
                if (frame.cols < cfg.tile_width || frame.rows < cfg.tile_height)
                    resize(frame, frame, Size(3840, 2160));
            }

            FrameMetrics fm;
            vector<Detection> dets = engine.run_frame(frame, i, fm, &logger);
            runs.push_back(fm);

            cout << "frame " << setw(4) << i
                 << " | " << fixed << setprecision(2) << setw(8) << fm.total_ms << " ms"
                 << " | " << setw(6) << fm.fps << " fps"
                 << " | cpu " << setw(6) << fm.cpu_percent << '%'
                 << " | gpu " << setw(6) << fm.gpu_percent << '%'
                 << " | rss " << setw(8) << fm.rss_mb << " MB"
                 << " | det " << fm.detections << endl;
        }

        print_summary(runs, engine.active_backend());
        cout << "[bench] Metrics written to benchmark_cpp_frames.csv / _tiles.csv" << endl;
    }
    catch (const std::exception& e) {
        cerr << "[fatal] " << e.what() << endl;
        return 1;
    }
    return 0;
}
#endif // BUILD_PYTHON_MODULE

// -----------------------------------------------------------------------------
// Pybind11 bridge - PyQt5 frontend hands 4K numpy arrays straight to C++
// -----------------------------------------------------------------------------
// The numpy buffer is wrapped, not copied, so no disk round-trip and no
// duplicate 24 MB allocation per frame. The GIL is released around inference
// so the PyQt5 event loop keeps repainting the visualiser.
#ifdef BUILD_PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

static Mat numpy_to_mat(py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& arr) {
    py::buffer_info info = arr.request();
    if (info.ndim != 3 || info.shape[2] != 3)
        throw runtime_error("Expected an HxWx3 uint8 BGR array.");
    return Mat(static_cast<int>(info.shape[0]),
               static_cast<int>(info.shape[1]),
               CV_8UC3, info.ptr);
}

PYBIND11_MODULE(yolo_backend_cpp, m) {
    m.doc() = "Test Path B - C++17 YOLOv8/SAHI inference engine";

    py::class_<Detection>(m, "Detection")
        .def_property_readonly("x", [](const Detection& d) { return d.box.x; })
        .def_property_readonly("y", [](const Detection& d) { return d.box.y; })
        .def_property_readonly("w", [](const Detection& d) { return d.box.width; })
        .def_property_readonly("h", [](const Detection& d) { return d.box.height; })
        .def_readonly("confidence", &Detection::confidence)
        .def_readonly("class_id",   &Detection::class_id)
        .def_readonly("tile_id",    &Detection::tile_id);

    py::class_<FrameMetrics>(m, "FrameMetrics")
        .def_readonly("frame_id",       &FrameMetrics::frame_id)
        .def_readonly("tiles",          &FrameMetrics::tiles)
        .def_readonly("preprocess_ms",  &FrameMetrics::preprocess_ms)
        .def_readonly("inference_ms",   &FrameMetrics::inference_ms)
        .def_readonly("postprocess_ms", &FrameMetrics::postprocess_ms)
        .def_readonly("slicing_ms",     &FrameMetrics::slicing_ms)
        .def_readonly("merge_ms",       &FrameMetrics::merge_ms)
        .def_readonly("total_ms",       &FrameMetrics::total_ms)
        .def_readonly("fps",            &FrameMetrics::fps)
        .def_readonly("cpu_percent",    &FrameMetrics::cpu_percent)
        .def_readonly("gpu_percent",    &FrameMetrics::gpu_percent)
        .def_readonly("gpu_mem_mb",     &FrameMetrics::gpu_mem_mb)
        .def_readonly("rss_mb",         &FrameMetrics::rss_mb)
        .def_readonly("peak_rss_mb",    &FrameMetrics::peak_rss_mb)
        .def_readonly("detections",     &FrameMetrics::detections);

    py::class_<YoloBackendCpp::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("model_path",     &YoloBackendCpp::Config::model_path)
        .def_readwrite("input_size",     &YoloBackendCpp::Config::input_size)
        .def_readwrite("use_cuda",       &YoloBackendCpp::Config::use_cuda)
        .def_readwrite("use_fp16",       &YoloBackendCpp::Config::use_fp16)
        .def_readwrite("conf_threshold", &YoloBackendCpp::Config::conf_threshold)
        .def_readwrite("nms_threshold",  &YoloBackendCpp::Config::nms_threshold)
        .def_readwrite("tile_width",     &YoloBackendCpp::Config::tile_width)
        .def_readwrite("tile_height",    &YoloBackendCpp::Config::tile_height)
        .def_readwrite("overlap_ratio",  &YoloBackendCpp::Config::overlap_ratio)
        .def_readwrite("warmup_runs",    &YoloBackendCpp::Config::warmup_runs);

    py::class_<YoloBackendCpp>(m, "YoloBackendCpp")
        .def(py::init<const YoloBackendCpp::Config&>())
        .def("warmup", &YoloBackendCpp::warmup,
             py::call_guard<py::gil_scoped_release>())
        .def("set_class_names", &YoloBackendCpp::set_class_names)
        .def_property_readonly("active_backend", &YoloBackendCpp::active_backend)
        .def("run_frame",
             [](YoloBackendCpp& self,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr,
                int frame_id) {
                 Mat frame = numpy_to_mat(arr);
                 FrameMetrics fm;
                 vector<Detection> dets;
                 {
                     py::gil_scoped_release release;
                     dets = self.run_frame(frame, frame_id, fm, nullptr);
                 }
                 return py::make_tuple(dets, fm);
             },
             py::arg("frame"), py::arg("frame_id") = 0,
             "Run SAHI-tiled inference on a 4K BGR frame. Returns (detections, metrics).");
}
#endif // BUILD_PYTHON_MODULE
