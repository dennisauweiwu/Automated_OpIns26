# =============================================================================
#  Test Path B build.
#
#  Two targets from one shared engine library:
#    aoi_engine_cpp   pybind11 extension, imported by aoi/engines.py
#    aoi_bench        standalone benchmark, no interpreter in the process
#
#  Configure (Windows / MSVC / x64):
#    cmake -B build -S . -A x64 ^
#          -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib" ^
#          -DONNXRUNTIME_ROOT="C:/onnxruntime-gpu" ^
#          -DPython_EXECUTABLE="C:/path/to/python.exe" ^
#          -DUSE_NVML=ON
#    cmake --build build --config Release
#
#  The extension MUST be built against the SAME interpreter that runs the GUI,
#  or `import aoi_engine_cpp` fails with an ABI error.
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(aoi_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/O2 /EHsc /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

option(USE_NVML "Enable NVML for GPU utilisation metrics" OFF)
option(BUILD_BENCH "Build the standalone benchmark executable" ON)

set(AOI_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/../aoi" CACHE PATH
    "Where the built extension and its runtime DLLs are placed")

# --- Dependencies ------------------------------------------------------------

# dnn is required for blobFromImage and NMSBoxesBatched only; inference itself
# runs on ONNX Runtime. Both routines are the exact ones the Python path calls,
# which is what keeps preprocessing and suppression identical across the paths.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
message(STATUS "OpenCV ${OpenCV_VERSION}")

set(ONNXRUNTIME_ROOT "" CACHE PATH
    "Extracted onnxruntime-win-x64-gpu-<version> folder (contains include/ and lib/)")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set. Download the Windows GPU archive from\n"
        "  https://github.com/microsoft/onnxruntime/releases\n"
        "extract it, then re-run cmake with -DONNXRUNTIME_ROOT=C:/onnxruntime-gpu")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_ROOT}/include.\n"
        "ONNXRUNTIME_ROOT must point at the folder that directly contains "
        "include/ and lib/.")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
             HINTS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# A CPU-only ORT archive builds and links exactly like the GPU one and then
# silently never touches the GPU - which would make the whole comparison a
# GPU-vs-CPU measurement reported as a language one. Catch it at configure time.
if(WIN32 AND NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
    message(WARNING
        "onnxruntime_providers_cuda.dll is NOT in ${ONNXRUNTIME_ROOT}/lib.\n"
        "This is the CPU-only package: the build will succeed and the engine "
        "will report 'ONNX Runtime / CPU'. Download 'onnxruntime-win-x64-gpu-*'.")
endif()

if(USE_NVML)
    find_path(NVML_INCLUDE_DIR nvml.h
              HINTS $ENV{CUDA_PATH}/include /usr/local/cuda/include)
    find_library(NVML_LIBRARY NAMES nvml nvidia-ml
                 HINTS $ENV{CUDA_PATH}/lib/x64 /usr/local/cuda/lib64/stubs)
    if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
        message(STATUS "NVML enabled: ${NVML_LIBRARY}")
    else()
        message(WARNING
            "NVML not found under CUDA_PATH='$ENV{CUDA_PATH}'. The engine will "
            "return -1 for GPU columns; aoi/engines.py fills those from pynvml, "
            "so this is a degradation, not a failure.")
        set(USE_NVML OFF)
    endif()
endif()

# --- Shared engine -----------------------------------------------------------
# Compiled once as an object library and linked into both targets, instead of
# each target rebuilding the same translation unit.

add_library(aoi_engine_obj OBJECT aoi_engine.cpp)
target_include_directories(aoi_engine_obj PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
    "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(aoi_engine_obj PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
if(WIN32)
    target_link_libraries(aoi_engine_obj PUBLIC psapi)
endif()
if(USE_NVML)
    target_compile_definitions(aoi_engine_obj PUBLIC USE_NVML)
    target_include_directories(aoi_engine_obj PUBLIC ${NVML_INCLUDE_DIR})
    target_link_libraries(aoi_engine_obj PUBLIC ${NVML_LIBRARY})
endif()

# --- Python extension --------------------------------------------------------

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RC)
    if(PYBIND11_RC EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Run: pip install pybind11")
    endif()
endif()

pybind11_add_module(aoi_engine_cpp bindings.cpp $<TARGET_OBJECTS:aoi_engine_obj>)
target_link_libraries(aoi_engine_cpp PRIVATE aoi_engine_obj)

# Drop the extension beside the Python package so `import aoi_engine_cpp` works
# without any sys.path juggling beyond what aoi/config.py::bootstrap already does.
set_target_properties(aoi_engine_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY         "${AOI_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${AOI_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY         "${AOI_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${AOI_PACKAGE_DIR}")

# --- Standalone benchmark ----------------------------------------------------

if(BUILD_BENCH)
    add_executable(aoi_bench bench_cli.cpp $<TARGET_OBJECTS:aoi_engine_obj>)
    target_link_libraries(aoi_bench PRIVATE aoi_engine_obj)
endif()

# --- Runtime DLLs beside the extension --------------------------------------
#
# OpenCV_DIR can legitimately be either the top-level "build" folder or the
# compiler-specific "build/x64/vcXX/lib" subfolder, so search the real layouts
# rather than assuming which one was passed.

if(WIN32)
    set(_ocv_dll
        "opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll")
    find_file(OPENCV_RUNTIME_DLL NAMES ${_ocv_dll}
        PATHS "${OpenCV_DIR}/bin"
              "${OpenCV_DIR}/../bin"
              "${OpenCV_DIR}/x64/vc17/bin"
              "${OpenCV_DIR}/x64/vc16/bin"
              "${OpenCV_DIR}/x64/vc15/bin"
        NO_DEFAULT_PATH)

    if(OPENCV_RUNTIME_DLL)
        add_custom_command(TARGET aoi_engine_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_RUNTIME_DLL}" "${AOI_PACKAGE_DIR}/"
            COMMENT "Copying OpenCV runtime next to the extension" VERBATIM)
    else()
        message(WARNING
            "Could not locate ${_ocv_dll}. The build will succeed but the import "
            "will fail until you copy it into ${AOI_PACKAGE_DIR}.")
    endif()

    # Every ORT dll goes across: onnxruntime.dll pulls in
    # onnxruntime_providers_shared.dll, which loads _cuda.dll when the session is
    # created. Missing any one degrades silently to CPU or fails the import.
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    if(ORT_DLLS)
        add_custom_command(TARGET aoi_engine_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_DLLS} "${AOI_PACKAGE_DIR}/"
            COMMENT "Copying ONNX Runtime next to the extension" VERBATIM)
    else()
        message(WARNING "No DLLs found in ${ONNXRUNTIME_ROOT}/lib")
    endif()
endif()

message(STATUS "Extension output: ${AOI_PACKAGE_DIR}")
