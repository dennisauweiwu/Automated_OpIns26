// =============================================================================
//  aoi_bench - standalone C++ benchmark, no Python in the process.
//
//  Why this exists alongside the Python-driven study: when Path B is called
//  through pybind11 it shares an address space with the interpreter, its
//  allocator and its GC. Anyone reviewing the results is entitled to ask
//  whether that contaminates the C++ numbers. This binary answers it - the same
//  engine, the same frames, no interpreter anywhere - so the two figures can be
//  compared and the bridge overhead quantified rather than assumed negligible.
//
//      aoi_bench --model weights.onnx --frames 30
//      aoi_bench --model weights.onnx --image board.png --csv out.csv
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "aoi_engine.hpp"

using namespace aoi;

namespace {

struct Args {
    std::string model;
    std::string image;
    std::string csv;
    int    frames    = 30;
    int    trials    = 3;
    int    discard   = 3;
    int    width     = 3840;
    int    height    = 2160;
    int    tile_w    = 1920;
    int    tile_h    = 1080;
    float  overlap   = 0.20f;
    float  conf      = 0.25f;
    bool   cpu_only  = false;
};

[[noreturn]] void usage(int code) {
    std::cout <<
        "aoi_bench - standalone C++ benchmark for Test Path B\n\n"
        "  --model PATH     ONNX model (required)\n"
        "  --image PATH     board image; omitted -> a synthetic gradient frame\n"
        "  --frames N       measured frames per trial   (default 30)\n"
        "  --trials N       independent trials          (default 3)\n"
        "  --discard N      settling frames per trial   (default 3)\n"
        "  --size WxH       input frame size            (default 3840x2160)\n"
        "  --tile WxH       SAHI tile size              (default 1920x1080)\n"
        "  --overlap F      requested overlap ratio     (default 0.20)\n"
        "  --conf F         confidence threshold        (default 0.25)\n"
        "  --cpu            disable the CUDA provider\n"
        "  --csv PATH       write per-frame rows\n";
    std::exit(code);
}

bool parse_pair(const std::string& s, int& a, int& b) {
    const auto x = s.find('x');
    if (x == std::string::npos) return false;
    a = std::stoi(s.substr(0, x));
    b = std::stoi(s.substr(x + 1));
    return true;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(2);
            return argv[++i];
        };
        if (k == "--model")        a.model = next();
        else if (k == "--image")   a.image = next();
        else if (k == "--csv")     a.csv = next();
        else if (k == "--frames")  a.frames = std::stoi(next());
        else if (k == "--trials")  a.trials = std::stoi(next());
        else if (k == "--discard") a.discard = std::stoi(next());
        else if (k == "--overlap") a.overlap = std::stof(next());
        else if (k == "--conf")    a.conf = std::stof(next());
        else if (k == "--cpu")     a.cpu_only = true;
        else if (k == "--size")  { if (!parse_pair(next(), a.width, a.height)) usage(2); }
        else if (k == "--tile")  { if (!parse_pair(next(), a.tile_w, a.tile_h)) usage(2); }
        else if (k == "-h" || k == "--help") usage(0);
        else { std::cerr << "unknown option: " << k << "\n"; usage(2); }
    }
    if (a.model.empty()) { std::cerr << "--model is required\n"; usage(2); }
    return a;
}

// Cheap stand-in when no board image is supplied. Deliberately not a PCB
// renderer: the C++ side has no business owning a second, divergent copy of
// aoi/synth.py. For a like-for-like input, pass --image with a frame exported
// by `python main.py board`.
cv::Mat synthetic(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        auto* row = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x)
            row[x] = cv::Vec3b(static_cast<uchar>((x * 7 + y * 3) & 0xFF),
                               static_cast<uchar>((x * 3 + y * 5) & 0xFF),
                               static_cast<uchar>((x + y) & 0xFF));
    }
    return img;
}

struct Stats {
    double mean = 0, sd = 0, p50 = 0, p95 = 0, lo = 0, hi = 0, cv = 0;
    size_t n = 0;
};

Stats describe(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n    = v.size();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0.0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.sd  = v.size() > 1 ? std::sqrt(acc / (v.size() - 1)) : 0.0;
    s.p50 = v[v.size() / 2];
    s.p95 = v[std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.95))];
    s.lo  = v.front();
    s.hi  = v.back();
    s.cv  = s.mean > 0 ? s.sd / s.mean : 0.0;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);

    Config cfg;
    cfg.model_path     = args.model;
    cfg.use_cuda       = !args.cpu_only;
    cfg.conf_threshold = args.conf;
    cfg.tile_width     = args.tile_w;
    cfg.tile_height    = args.tile_h;
    cfg.overlap_ratio  = args.overlap;

    cv::Mat frame;
    if (!args.image.empty()) {
        frame = cv::imread(args.image, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "could not read " << args.image << "\n";
            return 1;
        }
        if (frame.cols != args.width || frame.rows != args.height)
            cv::resize(frame, frame, cv::Size(args.width, args.height), 0, 0,
                       cv::INTER_AREA);
    } else {
        frame = synthetic(args.width, args.height);
    }

    std::cout << "======================================================\n"
              << "  aoi_bench - Test Path B, no interpreter in process\n"
              << "======================================================\n";

    try {
        AoiEngine engine(cfg);
        engine.warmup();

        const auto origins = engine.slice_origins(frame.size());
        std::cout << "  frame  " << frame.cols << "x" << frame.rows << "\n"
                  << "  tiles  " << origins.size() << " of " << args.tile_w
                  << "x" << args.tile_h << "\n"
                  << "  design " << args.trials << " trials x " << args.frames
                  << " frames, " << args.discard << " discarded each\n\n";

        std::ofstream csv;
        if (!args.csv.empty()) {
            csv.open(args.csv);
            csv << "trial,frame,total_ms,slice_ms,marshal_ms,preprocess_ms,"
                   "inference_ms,decode_ms,nms_ms,merge_ms,dispatch_ms,"
                   "cpu_percent,rss_mb,detections\n";
        }

        std::vector<double> all;
        std::vector<double> trial_means;
        FrameMetrics last{};

        for (int t = 0; t < args.trials; ++t) {
            std::vector<double> lat;
            for (int i = 0; i < args.frames; ++i) {
                FrameMetrics fm;
                auto dets = engine.run_frame(frame, i, fm);
                (void)dets;
                if (i < args.discard) continue;      // settle, do not record
                lat.push_back(fm.total_ms);
                last = fm;
                if (csv) {
                    csv << t << ',' << i << ',' << fm.total_ms << ','
                        << fm.slice_ms << ',' << fm.marshal_ms << ','
                        << fm.preprocess_ms << ',' << fm.inference_ms << ','
                        << fm.decode_ms << ',' << fm.nms_ms << ','
                        << fm.merge_ms << ',' << fm.dispatch_ms << ','
                        << fm.cpu_percent << ',' << fm.rss_mb << ','
                        << fm.detections << '\n';
                }
            }
            const Stats s = describe(lat);
            trial_means.push_back(s.mean);
            all.insert(all.end(), lat.begin(), lat.end());
            std::cout << "  trial " << (t + 1) << ": " << std::fixed
                      << std::setprecision(2) << s.mean << " ms  (CV "
                      << std::setprecision(1) << s.cv * 100.0 << "%)\n";
        }

        const Stats s = describe(all);
        std::cout << "\n------------------------------------------------------\n"
                  << std::fixed << std::setprecision(2)
                  << "  mean       " << s.mean << " ms\n"
                  << "  sd         " << s.sd << " ms\n"
                  << "  median     " << s.p50 << " ms\n"
                  << "  p95        " << s.p95 << " ms\n"
                  << "  min / max  " << s.lo << " / " << s.hi << " ms\n"
                  << "  CV         " << std::setprecision(1) << s.cv * 100.0 << " %\n"
                  << "  FPS        " << std::setprecision(2)
                  << (s.mean > 0 ? 1000.0 / s.mean : 0.0) << "\n"
                  << "  n          " << s.n << "\n";

        const Stats bt = describe(trial_means);
        std::cout << "  between-trial CV " << std::setprecision(1)
                  << bt.cv * 100.0 << " %"
                  << (bt.cv > 0.05 ? "   <- machine not settled; re-run" : "")
                  << "\n";

        std::cout << "\n  Stage means on the final frame (ms)\n"
                  << std::setprecision(3)
                  << "    slice      " << last.slice_ms << "\n"
                  << "    marshal    " << last.marshal_ms << "\n"
                  << "    preprocess " << last.preprocess_ms << "\n"
                  << "    inference  " << last.inference_ms << "\n"
                  << "    decode     " << last.decode_ms << "\n"
                  << "    nms        " << last.nms_ms << "\n"
                  << "    merge      " << last.merge_ms << "\n"
                  << "    dispatch   " << last.dispatch_ms << "\n"
                  << "\n  CPU " << std::setprecision(1) << last.cpu_percent
                  << " %   RSS " << std::setprecision(0) << last.rss_mb
                  << " MB   detections " << last.detections << "\n";

        if (!args.csv.empty())
            std::cout << "\n  per-frame rows -> " << args.csv << "\n";
    } catch (const std::exception& e) {
        std::cerr << "\n  FAILED: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
