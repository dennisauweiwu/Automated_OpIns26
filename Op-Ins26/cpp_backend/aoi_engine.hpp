// =============================================================================
//  Test Path B - C++17 inference engine, public interface.
//
//  Split from the implementation so the three consumers - the pybind11 module,
//  the standalone benchmark executable, and any future test harness - share one
//  declaration instead of each compiling a 1500-line translation unit that also
//  contains a main() behind an #ifdef.
//
//  PARITY CONTRACT
//  ---------------
//  The following must stay behaviourally identical to aoi/pipeline.py, or the
//  two test paths stop processing the same pixels and the comparison measures
//  nothing:
//
//      AoiEngine::axis_origins   <->  pipeline.axis_origins
//      AoiEngine::letterbox      <->  pipeline.letterbox
//      AoiEngine::decode         <->  engines.decode_yolov8
//      NMS parameters and order  <->  pipeline.global_nms / engines.tile_nms
//
//  Change one side and change the other in the same commit.
//
//  The FrameMetrics stage names match aoi.pipeline.STAGES exactly, so both
//  paths land in one table in the report with no translation layer.
// =============================================================================

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace aoi {

// -----------------------------------------------------------------------------
// Data contracts
// -----------------------------------------------------------------------------

struct Config {
    std::string model_path;
    int    input_size     = 640;    // fallback only; the model's own shape wins
    bool   use_cuda       = true;
    bool   use_fp16       = true;
    float  conf_threshold = 0.25f;
    float  nms_threshold  = 0.45f;
    int    tile_width     = 1920;
    int    tile_height    = 1080;
    float  overlap_ratio  = 0.20f;
    int    warmup_runs    = 5;
};

struct Detection {
    cv::Rect box;
    float    confidence = 0.f;
    int      class_id   = -1;
    int      tile_id    = -1;
};

// Field names mirror aoi.pipeline.STAGES. marshal_ms is the C++ counterpart of
// Python's cross-boundary pixel copy: here it is a cv::Mat ROI view, so it is
// expected to be near zero, and that near-zero IS the measurement the study is
// after. dispatch_ms is the residual - total minus every named stage - so
// framework overhead is reported explicitly instead of being absorbed into
// whichever stage happens to bracket it.
struct FrameMetrics {
    int    frame_id       = 0;
    int    tiles          = 0;
    double slice_ms       = 0.0;
    double marshal_ms     = 0.0;
    double preprocess_ms  = 0.0;
    double inference_ms   = 0.0;
    double decode_ms      = 0.0;
    double nms_ms         = 0.0;
    double merge_ms       = 0.0;
    double dispatch_ms    = 0.0;
    double total_ms       = 0.0;
    double fps            = 0.0;
    double cpu_percent    = 0.0;
    double gpu_percent    = -1.0;
    double gpu_mem_mb     = -1.0;
    double rss_mb         = 0.0;
    double peak_rss_mb    = 0.0;
    int    detections     = 0;
};

// -----------------------------------------------------------------------------
// SystemMonitor
// -----------------------------------------------------------------------------

// CPU is normalised by logical core count, matching HostMonitor in
// aoi/engines.py: 100% means every core busy on both paths, rather than 1200%
// on one and 100% on the other.
class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    void   reset_cpu_baseline();
    double cpu_percent();
    double rss_mb() const;
    double peak_rss_mb() const;
    double gpu_percent() const;      // -1 when built without USE_NVML
    double gpu_mem_mb() const;       // -1 when built without USE_NVML
    std::string device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------------
// AoiEngine
// -----------------------------------------------------------------------------

class AoiEngine {
public:
    explicit AoiEngine(const Config& cfg);
    ~AoiEngine();

    AoiEngine(const AoiEngine&)            = delete;
    AoiEngine& operator=(const AoiEngine&) = delete;

    // The first forward on a CUDA target pays for cuDNN algorithm search and
    // workspace allocation. Excluding it keeps the comparison honest -
    // Ultralytics warms up internally too.
    void warmup();

    std::vector<Detection> run_frame(const cv::Mat& frame, int frame_id,
                                     FrameMetrics& out);

    void set_class_names(const std::vector<std::string>& names);
    const std::vector<std::string>& class_names() const;

    const std::string& backend_description() const;
    std::string        device_name() const;
    cv::Size           model_input() const;

    // ---- geometry, mirrored in aoi/pipeline.py --------------------------
    static std::vector<int> axis_origins(int total, int tile, float overlap);
    std::vector<cv::Point>  slice_origins(const cv::Size& frame) const;

    static cv::Mat letterbox(const cv::Mat& src, const cv::Size& shape,
                             float& scale, int& pad_x, int& pad_y);

    // Providers this ONNX Runtime build actually registered. Reported by the
    // doctor command, because a CPU-only ORT package links and runs exactly
    // like the GPU one and then silently never touches the GPU.
    static std::vector<std::string> available_providers();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------------
// Small timing helper - one place that decides what a millisecond is.
// -----------------------------------------------------------------------------

class Stopwatch {
public:
    Stopwatch() : t0_(clock::now()) {}
    void reset() { t0_ = clock::now(); }
    double lap() {
        const auto now = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t0_).count();
        t0_ = now;
        return ms;
    }
    double peek() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_;
};

}  // namespace aoi
