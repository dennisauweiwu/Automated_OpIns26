// =============================================================================
//  pybind11 bridge for Test Path B.
//
//  Module name: aoi_engine_cpp   (imported by aoi/engines.py::CppEngine)
//
//  The numpy buffer is WRAPPED as a cv::Mat, not copied: a 4K BGR frame is
//  24 MB, and copying it per call would put the bridge's own cost into the
//  measurement it exists to make. The GIL is released around run_frame so the
//  PyQt5 event loop keeps repainting while inference runs - without that, the
//  C++ path would appear to freeze the interface that the Python path does not.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <stdexcept>

#include "aoi_engine.hpp"

namespace py = pybind11;
using namespace aoi;

namespace {

cv::Mat numpy_to_mat(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& arr) {
    py::buffer_info info = arr.request();
    if (info.ndim != 3 || info.shape[2] != 3)
        throw std::runtime_error("Expected an HxWx3 uint8 BGR array.");
    // No copy: the Mat header points at numpy's buffer. Valid only while the
    // caller holds a reference to the array, which run_frame's signature
    // guarantees for the duration of the call.
    return cv::Mat(static_cast<int>(info.shape[0]),
                   static_cast<int>(info.shape[1]),
                   CV_8UC3, info.ptr);
}

}  // namespace

PYBIND11_MODULE(aoi_engine_cpp, m) {
    m.doc() = "Test Path B - C++17 YOLOv8 + SAHI engine on ONNX Runtime";
    m.attr("__version__") = "2.0.0";

    m.def("available_providers", &AoiEngine::available_providers,
          "Execution providers this ONNX Runtime build registered. A CPU-only "
          "package links and runs exactly like the GPU one, so this is how you "
          "find out which you have.");

    m.def("axis_origins", &AoiEngine::axis_origins,
          py::arg("total"), py::arg("tile"), py::arg("overlap"),
          "Tile origins along one axis. Exposed so a test can assert that this "
          "returns exactly what aoi.pipeline.axis_origins returns - the parity "
          "contract the whole comparison rests on.");

    py::class_<Detection>(m, "Detection")
        .def_property_readonly("x", [](const Detection& d) { return d.box.x; })
        .def_property_readonly("y", [](const Detection& d) { return d.box.y; })
        .def_property_readonly("w", [](const Detection& d) { return d.box.width; })
        .def_property_readonly("h", [](const Detection& d) { return d.box.height; })
        .def_readonly("confidence", &Detection::confidence)
        .def_readonly("class_id",   &Detection::class_id)
        .def_readonly("tile_id",    &Detection::tile_id)
        .def("__repr__", [](const Detection& d) {
            return "<Detection cls=" + std::to_string(d.class_id) +
                   " conf=" + std::to_string(d.confidence) +
                   " box=(" + std::to_string(d.box.x) + "," +
                   std::to_string(d.box.y) + "," +
                   std::to_string(d.box.width) + "," +
                   std::to_string(d.box.height) + ")>";
        });

    // Field names match aoi.pipeline.STAGES so both paths land in one table.
    py::class_<FrameMetrics>(m, "FrameMetrics")
        .def_readonly("frame_id",      &FrameMetrics::frame_id)
        .def_readonly("tiles",         &FrameMetrics::tiles)
        .def_readonly("slice_ms",      &FrameMetrics::slice_ms)
        .def_readonly("marshal_ms",    &FrameMetrics::marshal_ms)
        .def_readonly("preprocess_ms", &FrameMetrics::preprocess_ms)
        .def_readonly("inference_ms",  &FrameMetrics::inference_ms)
        .def_readonly("decode_ms",     &FrameMetrics::decode_ms)
        .def_readonly("nms_ms",        &FrameMetrics::nms_ms)
        .def_readonly("merge_ms",      &FrameMetrics::merge_ms)
        .def_readonly("dispatch_ms",   &FrameMetrics::dispatch_ms)
        .def_readonly("total_ms",      &FrameMetrics::total_ms)
        .def_readonly("fps",           &FrameMetrics::fps)
        .def_readonly("cpu_percent",   &FrameMetrics::cpu_percent)
        .def_readonly("gpu_percent",   &FrameMetrics::gpu_percent)
        .def_readonly("gpu_mem_mb",    &FrameMetrics::gpu_mem_mb)
        .def_readonly("rss_mb",        &FrameMetrics::rss_mb)
        .def_readonly("peak_rss_mb",   &FrameMetrics::peak_rss_mb)
        .def_readonly("detections",    &FrameMetrics::detections);

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("model_path",     &Config::model_path)
        .def_readwrite("input_size",     &Config::input_size)
        .def_readwrite("use_cuda",       &Config::use_cuda)
        .def_readwrite("use_fp16",       &Config::use_fp16)
        .def_readwrite("conf_threshold", &Config::conf_threshold)
        .def_readwrite("nms_threshold",  &Config::nms_threshold)
        .def_readwrite("tile_width",     &Config::tile_width)
        .def_readwrite("tile_height",    &Config::tile_height)
        .def_readwrite("overlap_ratio",  &Config::overlap_ratio)
        .def_readwrite("warmup_runs",    &Config::warmup_runs);

    py::class_<AoiEngine>(m, "AoiEngine")
        .def(py::init<const Config&>())
        .def("warmup", &AoiEngine::warmup, py::call_guard<py::gil_scoped_release>())
        .def("set_class_names", &AoiEngine::set_class_names)
        .def_property_readonly("backend_description",
                               &AoiEngine::backend_description)
        .def_property_readonly("device_name", &AoiEngine::device_name)
        .def_property_readonly("model_input", [](const AoiEngine& e) {
            const auto s = e.model_input();
            return py::make_tuple(s.width, s.height);
        })
        .def("slice_origins", [](const AoiEngine& e, int w, int h) {
            std::vector<std::pair<int, int>> out;
            for (const auto& p : e.slice_origins(cv::Size(w, h)))
                out.emplace_back(p.x, p.y);
            return out;
        }, py::arg("width"), py::arg("height"))
        .def("run_frame",
             [](AoiEngine& self,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr,
                int frame_id) {
                 cv::Mat frame = numpy_to_mat(arr);
                 FrameMetrics fm;
                 std::vector<Detection> dets;
                 {
                     py::gil_scoped_release release;
                     dets = self.run_frame(frame, frame_id, fm);
                 }
                 return py::make_tuple(dets, fm);
             },
             py::arg("frame"), py::arg("frame_id") = 0,
             "SAHI-tiled inference on a 4K BGR frame. Returns (detections, metrics).");
}
