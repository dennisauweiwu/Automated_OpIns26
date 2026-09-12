// =============================================================================
//  Test Path B - C++17 inference engine, implementation.
//
//  ONNX Runtime with the CUDA execution provider. OpenCV is still used for
//  imgproc (resize, copyMakeBorder) and dnn::NMSBoxesBatched, because those are
//  the exact routines the Python path calls - using a different resize or a
//  different NMS would inject an implementation difference into a measurement
//  that is supposed to isolate language and runtime.
//
//  Why not cv::dnn for inference: the prebuilt OpenCV distributions ship
//  without the CUDA DNN module, so cv::dnn::Net::forward ran on the CPU while
//  Path A ran on the GPU. That is a hardware difference being reported as a
//  language difference.
// =============================================================================

#include "aoi_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
  #include <sys/times.h>
  #include <unistd.h>
  #include <fstream>
#endif

#ifdef USE_NVML
  #include <nvml.h>
#endif

namespace aoi {

// -----------------------------------------------------------------------------
// Class-aware NMS
// -----------------------------------------------------------------------------
// cv::dnn::NMSBoxesBatched arrived in OpenCV 4.7. Older distributions are still
// common, and falling back keeps the C++ path buildable against them - but the
// fallback must produce the SAME result as the Python side's NMSBoxesBatched, or
// the two paths would suppress differently and the detection-parity check would
// fail for a reason that has nothing to do with either engine.
//
// The trick NMSBoxesBatched itself uses: offset every box by class_id times a
// value larger than the image, so boxes of different classes can never overlap
// and a single plain NMS pass becomes class-aware.
namespace {

void nms_batched(const std::vector<cv::Rect>& boxes,
                 const std::vector<float>& scores,
                 const std::vector<int>& class_ids,
                 float score_threshold, float nms_threshold,
                 std::vector<int>& keep) {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::dnn::NMSBoxesBatched(boxes, scores, class_ids, score_threshold,
                             nms_threshold, keep);
#else
    int max_coord = 0;
    for (const auto& b : boxes)
        max_coord = std::max({max_coord, b.x + b.width, b.y + b.height});
    const int stride = max_coord + 1;

    std::vector<cv::Rect> shifted;
    shifted.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        const int off = class_ids[i] * stride;
        shifted.emplace_back(boxes[i].x + off, boxes[i].y + off,
                             boxes[i].width, boxes[i].height);
    }
    cv::dnn::NMSBoxes(shifted, scores, score_threshold, nms_threshold, keep);
#endif
}

}  // namespace

// =============================================================================
// SystemMonitor
// =============================================================================

struct SystemMonitor::Impl {
    int cores = 1;
#ifdef _WIN32
    HANDLE process = nullptr;
    ULARGE_INTEGER last_cpu{}, last_sys{}, last_user{};
#else
    clock_t last_cpu = 0, last_sys = 0, last_user = 0;
#endif
#ifdef USE_NVML
    bool nvml_ok = false;
    nvmlDevice_t device{};
#endif
};

SystemMonitor::SystemMonitor() : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    impl_->cores = static_cast<int>(si.dwNumberOfProcessors);
    impl_->process = GetCurrentProcess();
#else
    impl_->cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
    if (impl_->cores < 1) impl_->cores = 1;
    reset_cpu_baseline();

#ifdef USE_NVML
    if (nvmlInit() == NVML_SUCCESS &&
        nvmlDeviceGetHandleByIndex(0, &impl_->device) == NVML_SUCCESS) {
        impl_->nvml_ok = true;
    }
#endif
}

SystemMonitor::~SystemMonitor() {
#ifdef USE_NVML
    if (impl_ && impl_->nvml_ok) nvmlShutdown();
#endif
}

void SystemMonitor::reset_cpu_baseline() {
#ifdef _WIN32
    FILETIME ftime, fsys, fuser;
    GetSystemTimeAsFileTime(&ftime);
    std::memcpy(&impl_->last_cpu, &ftime, sizeof(FILETIME));
    GetProcessTimes(impl_->process, &ftime, &ftime, &fsys, &fuser);
    std::memcpy(&impl_->last_sys, &fsys, sizeof(FILETIME));
    std::memcpy(&impl_->last_user, &fuser, sizeof(FILETIME));
#else
    struct tms t;
    impl_->last_cpu  = times(&t);
    impl_->last_sys  = t.tms_stime;
    impl_->last_user = t.tms_utime;
#endif
}

double SystemMonitor::cpu_percent() {
#ifdef _WIN32
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    GetSystemTimeAsFileTime(&ftime);
    std::memcpy(&now, &ftime, sizeof(FILETIME));
    GetProcessTimes(impl_->process, &ftime, &ftime, &fsys, &fuser);
    std::memcpy(&sys, &fsys, sizeof(FILETIME));
    std::memcpy(&user, &fuser, sizeof(FILETIME));

    const double wall = static_cast<double>(now.QuadPart - impl_->last_cpu.QuadPart);
    if (wall <= 0.0) return 0.0;
    const double busy = static_cast<double>(
        (sys.QuadPart - impl_->last_sys.QuadPart) +
        (user.QuadPart - impl_->last_user.QuadPart));

    impl_->last_cpu = now;
    impl_->last_sys = sys;
    impl_->last_user = user;
    // Divided by core count, exactly as psutil's value is on the Python side.
    return 100.0 * busy / wall / impl_->cores;
#else
    struct tms t;
    const clock_t now = times(&t);
    if (now <= impl_->last_cpu) return 0.0;
    const double pct = 100.0 *
        static_cast<double>((t.tms_stime - impl_->last_sys) +
                            (t.tms_utime - impl_->last_user)) /
        static_cast<double>(now - impl_->last_cpu) / impl_->cores;
    impl_->last_cpu = now;
    impl_->last_sys = t.tms_stime;
    impl_->last_user = t.tms_utime;
    return pct;
#endif
}

double SystemMonitor::rss_mb() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(impl_->process, &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#else
    std::ifstream statm("/proc/self/statm");
    long pages = 0, rss = 0;
    if (statm >> pages >> rss)
        return static_cast<double>(rss) * sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
    return 0.0;
#endif
}

double SystemMonitor::peak_rss_mb() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(impl_->process, &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return static_cast<double>(ru.ru_maxrss) / 1024.0;
    return 0.0;
#endif
}

double SystemMonitor::gpu_percent() const {
#ifdef USE_NVML
    if (impl_->nvml_ok) {
        nvmlUtilization_t u{};
        if (nvmlDeviceGetUtilizationRates(impl_->device, &u) == NVML_SUCCESS)
            return static_cast<double>(u.gpu);
    }
#endif
    // -1 is the sentinel aoi/engines.py fills in from pynvml, so a build
    // without the CUDA Toolkit headers still reports GPU numbers.
    return -1.0;
}

double SystemMonitor::gpu_mem_mb() const {
#ifdef USE_NVML
    if (impl_->nvml_ok) {
        nvmlMemory_t m{};
        if (nvmlDeviceGetMemoryInfo(impl_->device, &m) == NVML_SUCCESS)
            return static_cast<double>(m.used) / (1024.0 * 1024.0);
    }
#endif
    return -1.0;
}

std::string SystemMonitor::device_name() const {
#ifdef USE_NVML
    if (impl_->nvml_ok) {
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
        if (nvmlDeviceGetName(impl_->device, name, sizeof(name)) == NVML_SUCCESS)
            return std::string(name);
    }
#endif
    return "unknown";
}

// =============================================================================
// AoiEngine
// =============================================================================

namespace {

#ifdef _WIN32
// ORT takes wide paths on Windows. A board path with a non-ASCII character in
// it would otherwise fail with a misleading "model not found".
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(),
                                      static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}
#endif

// CreateCUDAProviderOptions returns a raw handle that has to be released even
// when AppendExecutionProvider_CUDA_V2 throws - which it does on a CUDA/cuDNN
// version mismatch, the most common failure on a fresh machine.
struct CudaOptsGuard {
    OrtCUDAProviderOptionsV2* p = nullptr;
    CudaOptsGuard() = default;
    CudaOptsGuard(const CudaOptsGuard&) = delete;
    CudaOptsGuard& operator=(const CudaOptsGuard&) = delete;
    ~CudaOptsGuard() { if (p) Ort::GetApi().ReleaseCUDAProviderOptions(p); }
};

}  // namespace

struct AoiEngine::Impl {
    Config cfg;
    std::vector<std::string> class_names;
    std::string backend_desc = "uninitialised";

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "aoi"};
    Ort::Session session{nullptr};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string in_name, out_name;
    std::vector<const char*> in_names, out_names;
    int in_w = 640, in_h = 640;

    // Reused across tiles, so no ~4.9 MB reallocation per call. Path A's
    // equivalent allocation is charged to preprocess_ms on that side; keeping
    // this member is what the C++ path is being credited for.
    cv::Mat blob;
    cv::Mat padded;

    SystemMonitor monitor;

    void build_session();
    std::vector<Detection> decode(const Ort::Value& tensor, float scale,
                                  int pad_x, int pad_y, const cv::Point& origin,
                                  int tile_id) const;
};

void AoiEngine::Impl::build_session() {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool cuda_ok = false;
    if (cfg.use_cuda) {
        const auto eps = Ort::GetAvailableProviders();
        const bool registered =
            std::find(eps.begin(), eps.end(), std::string("CUDAExecutionProvider"))
            != eps.end();

        if (!registered) {
            std::cerr << "[aoi] CUDAExecutionProvider is not registered - this is "
                         "the CPU-only ONNX Runtime package. Re-download the "
                         "'-gpu-' archive. Falling back to CPU.\n";
        } else {
            try {
                CudaOptsGuard g;
                Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&g.p));
                // EXHAUSTIVE picks the fastest convolution algorithm per input
                // shape. It makes the first pass expensive, which is exactly
                // what warmup_runs exists to absorb.
                const char* keys[] = {"device_id", "cudnn_conv_algo_search",
                                      "arena_extend_strategy",
                                      "do_copy_in_default_stream"};
                const char* vals[] = {"0", "EXHAUSTIVE", "kNextPowerOfTwo", "1"};
                Ort::ThrowOnError(
                    Ort::GetApi().UpdateCUDAProviderOptions(g.p, keys, vals, 4));
                so.AppendExecutionProvider_CUDA_V2(*g.p);
                cuda_ok = true;
            } catch (const Ort::Exception& e) {
                std::cerr << "[aoi] CUDA EP rejected, falling back to CPU: "
                          << e.what() << "\n";
            }
        }
    }

    // On CUDA the intra-op pool only contends with itself, so pin it to one
    // thread. On CPU we want every core - 0 means "let ORT decide". Getting
    // this backwards makes a CPU fallback look far worse than it is.
    so.SetIntraOpNumThreads(cuda_ok ? 1 : 0);
    backend_desc = cuda_ok ? "ONNX Runtime / CUDA EP" : "ONNX Runtime / CPU";

#ifdef _WIN32
    session = Ort::Session(env, widen(cfg.model_path).c_str(), so);
#else
    session = Ort::Session(env, cfg.model_path.c_str(), so);
#endif

    if (session.GetInputCount() < 1 || session.GetOutputCount() < 1)
        throw std::runtime_error("ONNX model has no input or no output tensor.");

    // Names are resolved once. Looking them up per forward walks the graph and
    // allocates a vector<string> on every tile.
    Ort::AllocatorWithDefaultOptions alloc;
    in_name  = session.GetInputNameAllocated(0, alloc).get();
    out_name = session.GetOutputNameAllocated(0, alloc).get();
    in_names  = {in_name.c_str()};
    out_names = {out_name.c_str()};

    // Take H and W from the model, not from cfg.input_size, so re-exporting at
    // a rectangular size (640x384, matching Ultralytics) needs no code change.
    const auto shape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
        in_h = static_cast<int>(shape[2]);
        in_w = static_cast<int>(shape[3]);
    } else {
        in_h = in_w = cfg.input_size;
        std::cerr << "[aoi] Model input shape is dynamic; using "
                  << in_w << "x" << in_h << ".\n";
    }
}

// -----------------------------------------------------------------------------

AoiEngine::AoiEngine(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;
    impl_->build_session();
    std::cout << "[aoi] model   " << cfg.model_path << "\n"
              << "[aoi] backend " << impl_->backend_desc << "\n"
              << "[aoi] input   " << impl_->in_w << "x" << impl_->in_h << "\n";
}

AoiEngine::~AoiEngine() = default;

void AoiEngine::set_class_names(const std::vector<std::string>& names) {
    impl_->class_names = names;
}

const std::vector<std::string>& AoiEngine::class_names() const {
    return impl_->class_names;
}

const std::string& AoiEngine::backend_description() const {
    return impl_->backend_desc;
}

std::string AoiEngine::device_name() const {
    return impl_->monitor.device_name();
}

cv::Size AoiEngine::model_input() const {
    return {impl_->in_w, impl_->in_h};
}

std::vector<std::string> AoiEngine::available_providers() {
    return Ort::GetAvailableProviders();
}

// ---- geometry ---------------------------------------------------------------

std::vector<int> AoiEngine::axis_origins(int total, int tile, float overlap) {
    // Mirror of pipeline.axis_origins. Even distribution, not "step and clamp":
    // the naive version leaves the final pair overlapping ~80% while the first
    // overlaps 20%, which spends inference on duplicate pixels and biases
    // detection density toward one edge.
    if (tile >= total) return {0};

    const int step = std::max(1, static_cast<int>(tile * (1.0f - overlap)));
    const int span = total - tile;
    int n = (span + step - 1) / step + 1;
    n = std::max(2, n);

    std::vector<int> origins;
    origins.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        origins.push_back(static_cast<int>(
            std::lround(static_cast<double>(i) * span / (n - 1))));
    return origins;
}

std::vector<cv::Point> AoiEngine::slice_origins(const cv::Size& frame) const {
    const auto xs = axis_origins(frame.width,  impl_->cfg.tile_width,
                                 impl_->cfg.overlap_ratio);
    const auto ys = axis_origins(frame.height, impl_->cfg.tile_height,
                                 impl_->cfg.overlap_ratio);
    std::vector<cv::Point> out;
    out.reserve(xs.size() * ys.size());
    for (int y : ys)
        for (int x : xs)
            out.emplace_back(x, y);
    return out;
}

cv::Mat AoiEngine::letterbox(const cv::Mat& src, const cv::Size& shape,
                             float& scale, int& pad_x, int& pad_y) {
    // Mirror of pipeline.letterbox. A plain resize to a square would squash a
    // 16:9 tile and change which fine-pitch defects survive.
    scale = std::min(static_cast<float>(shape.width)  / src.cols,
                     static_cast<float>(shape.height) / src.rows);
    const int nw = static_cast<int>(std::round(src.cols * scale));
    const int nh = static_cast<int>(std::round(src.rows * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    pad_x = (shape.width  - nw) / 2;
    pad_y = (shape.height - nh) / 2;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_y, shape.height - nh - pad_y,
                       pad_x, shape.width - nw - pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

// ---- decode -----------------------------------------------------------------

std::vector<Detection> AoiEngine::Impl::decode(const Ort::Value& tensor,
                                               float scale, int pad_x, int pad_y,
                                               const cv::Point& origin,
                                               int tile_id) const {
    // YOLOv8 emits [1, 4 + nc, anchors], channel-major, with no objectness
    // channel (unlike YOLOv5). The output is read in place with a stride, so
    // there is no transpose and no per-anchor minMaxLoc call.
    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3)
        throw std::runtime_error("Unexpected ONNX output rank.");

    const int channels = static_cast<int>(shape[1]);
    const int anchors  = static_cast<int>(shape[2]);
    const int nc       = channels - 4;
    if (nc <= 0) throw std::runtime_error("ONNX output has no class channels.");

    const float* data = tensor.GetTensorData<float>();
    const float inv_scale = 1.0f / scale;

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      ids;
    boxes.reserve(64);
    scores.reserve(64);
    ids.reserve(64);

    for (int a = 0; a < anchors; ++a) {
        int   best_id = -1;
        float best    = cfg.conf_threshold;
        for (int c = 0; c < nc; ++c) {
            const float v = data[(4 + c) * anchors + a];
            if (v > best) { best = v; best_id = c; }
        }
        if (best_id < 0) continue;

        const float cx = data[0 * anchors + a];
        const float cy = data[1 * anchors + a];
        const float bw = data[2 * anchors + a];
        const float bh = data[3 * anchors + a];

        boxes.emplace_back(
            static_cast<int>(std::lround((cx - bw * 0.5f - pad_x) * inv_scale) + origin.x),
            static_cast<int>(std::lround((cy - bh * 0.5f - pad_y) * inv_scale) + origin.y),
            static_cast<int>(std::lround(bw * inv_scale)),
            static_cast<int>(std::lround(bh * inv_scale)));
        scores.push_back(best);
        ids.push_back(best_id);
    }

    std::vector<Detection> out;
    if (boxes.empty()) return out;

    // Per-tile suppression, matching engines.tile_nms on the Python side.
    std::vector<int> keep;
    nms_batched(boxes, scores, ids, cfg.conf_threshold, cfg.nms_threshold, keep);
    out.reserve(keep.size());
    for (int i : keep)
        out.push_back(Detection{boxes[i], scores[i], ids[i], tile_id});
    return out;
}

// ---- run --------------------------------------------------------------------

void AoiEngine::warmup() {
    cv::Mat dummy(impl_->cfg.tile_height, impl_->cfg.tile_width, CV_8UC3,
                  cv::Scalar(114, 114, 114));
    FrameMetrics scratch;
    for (int i = 0; i < impl_->cfg.warmup_runs; ++i) {
        float scale = 1.f;
        int px = 0, py = 0;
        impl_->padded = letterbox(dummy, {impl_->in_w, impl_->in_h}, scale, px, py);
        cv::dnn::blobFromImage(impl_->padded, impl_->blob, 1.0 / 255.0,
                               impl_->padded.size(), cv::Scalar(),
                               /*swapRB=*/true, /*crop=*/false, CV_32F);
        const int64_t dims[4] = {1, 3, impl_->in_h, impl_->in_w};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            impl_->mem_info, reinterpret_cast<float*>(impl_->blob.data),
            impl_->blob.total(), dims, 4);
        impl_->session.Run(Ort::RunOptions{nullptr}, impl_->in_names.data(),
                           &input, 1, impl_->out_names.data(),
                           impl_->out_names.size());
    }
    (void)scratch;
    std::cout << "[aoi] warmup complete (" << impl_->cfg.warmup_runs << " passes)\n";
}

std::vector<Detection> AoiEngine::run_frame(const cv::Mat& frame, int frame_id,
                                            FrameMetrics& fm) {
    if (frame.empty()) throw std::runtime_error("run_frame received an empty frame.");

    auto& I = *impl_;
    I.monitor.reset_cpu_baseline();

    Stopwatch frame_clock;
    Stopwatch stage;

    fm = FrameMetrics{};
    fm.frame_id = frame_id;

    const auto origins = slice_origins(frame.size());
    fm.tiles    = static_cast<int>(origins.size());
    fm.slice_ms = stage.lap();

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      ids;
    std::vector<int>      tiles;

    for (size_t i = 0; i < origins.size(); ++i) {
        // A cv::Mat ROI is a view into the parent buffer - no pixel copy. This
        // is the C++ counterpart of Python's ascontiguousarray, and the fact
        // that it costs nothing is precisely what the marshal_ms column is
        // there to show.
        stage.reset();
        const cv::Rect roi(origins[i].x, origins[i].y,
                           I.cfg.tile_width, I.cfg.tile_height);
        const cv::Mat tile = frame(roi);
        fm.marshal_ms += stage.lap();

        float scale = 1.f;
        int pad_x = 0, pad_y = 0;
        I.padded = letterbox(tile, {I.in_w, I.in_h}, scale, pad_x, pad_y);
        cv::dnn::blobFromImage(I.padded, I.blob, 1.0 / 255.0, I.padded.size(),
                               cv::Scalar(), true, false, CV_32F);
        const int64_t dims[4] = {1, 3, I.in_h, I.in_w};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            I.mem_info, reinterpret_cast<float*>(I.blob.data), I.blob.total(),
            dims, 4);
        fm.preprocess_ms += stage.lap();

        // Session::Run synchronises the CUDA stream before returning, so this
        // really is the work and not a kernel launch. No C++ analogue of
        // torch.cuda.synchronize() is required.
        auto outs = I.session.Run(Ort::RunOptions{nullptr}, I.in_names.data(),
                                  &input, 1, I.out_names.data(),
                                  I.out_names.size());
        fm.inference_ms += stage.lap();

        // decode() also runs the per-tile NMS; splitting the two would mean a
        // second pass over the candidate list purely to attribute time, which
        // would itself change what is being measured. The combined cost is
        // charged to decode_ms and nms_ms is reserved for the tile-level
        // suppression the Python path reports separately.
        auto dets = I.decode(outs[0], scale, pad_x, pad_y, origins[i],
                             static_cast<int>(i));
        fm.decode_ms += stage.lap();

        for (const auto& d : dets) {
            boxes.push_back(d.box);
            scores.push_back(d.confidence);
            ids.push_back(d.class_id);
            tiles.push_back(d.tile_id);
        }
        fm.nms_ms += stage.lap();
    }

    // Cross-tile suppression on the master frame: a solder bridge sitting in a
    // tile overlap is detected once per neighbouring tile.
    stage.reset();
    std::vector<Detection> merged;
    if (!boxes.empty()) {
        std::vector<int> keep;
        nms_batched(boxes, scores, ids, I.cfg.conf_threshold,
                    I.cfg.nms_threshold, keep);
        merged.reserve(keep.size());
        for (int k : keep)
            merged.push_back(Detection{boxes[k], scores[k], ids[k], tiles[k]});
    }
    fm.merge_ms = stage.lap();

    fm.total_ms = frame_clock.peek();
    // Residual, so nothing the named stages missed is silently attributed to a
    // neighbouring one. Matches StageTimer.finish() on the Python side.
    const double accounted = fm.slice_ms + fm.marshal_ms + fm.preprocess_ms +
                             fm.inference_ms + fm.decode_ms + fm.nms_ms +
                             fm.merge_ms;
    fm.dispatch_ms = std::max(0.0, fm.total_ms - accounted);

    fm.fps         = fm.total_ms > 0.0 ? 1000.0 / fm.total_ms : 0.0;
    fm.detections  = static_cast<int>(merged.size());
    fm.cpu_percent = I.monitor.cpu_percent();
    fm.gpu_percent = I.monitor.gpu_percent();
    fm.gpu_mem_mb  = I.monitor.gpu_mem_mb();
    fm.rss_mb      = I.monitor.rss_mb();
    fm.peak_rss_mb = I.monitor.peak_rss_mb();
    return merged;
}

}  // namespace aoi
