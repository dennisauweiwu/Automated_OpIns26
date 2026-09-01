# =============================================================================
#  Test Path B build - produces BOTH:
#    1. aoi_cpp_bench.exe    -> standalone C++ benchmark (no Python in the loop)
#    2. yolo_backend_cpp.pyd -> Python extension imported by the PyQt5 GUI
#
#  Configure (Windows / MSVC / x64):
#    cmake -B build -S . -A x64 ^
#          -DOpenCV_DIR="C:/opencv/build" ^
#          -DPython_EXECUTABLE="C:/Users/<you>/venv/Scripts/python.exe" ^
#          -DUSE_NVML=ON
#    cmake --build build --config Release
#
#  The .pyd MUST be built against the SAME Python interpreter that runs the GUI,
#  otherwise `import yolo_backend_cpp` fails with a DLL/ABI error.
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(aoi_cpp_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/O2 /EHsc /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-O3 -Wall)
endif()

option(USE_NVML "Enable Nvidia NVML for GPU utilisation metrics" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn highgui videoio)
message(STATUS "OpenCV ${OpenCV_VERSION} -> ${OpenCV_INCLUDE_DIRS}")

# --- NVML (optional) ---------------------------------------------------------
if(USE_NVML)
    find_path(NVML_INCLUDE_DIR nvml.h
        HINTS $ENV{CUDA_PATH}/include /usr/local/cuda/include)
    find_library(NVML_LIBRARY NAMES nvml nvidia-ml
        HINTS $ENV{CUDA_PATH}/lib/x64 /usr/local/cuda/lib64/stubs)
    if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
        message(STATUS "NVML enabled: ${NVML_LIBRARY}")
    else()
        message(WARNING "NVML not found - GPU metrics will report -1.")
        set(USE_NVML OFF)
    endif()
endif()

# --- 1. Standalone benchmark executable --------------------------------------
add_executable(aoi_cpp_bench yolo_backend_cpp.cpp)
target_link_libraries(aoi_cpp_bench PRIVATE ${OpenCV_LIBS})
target_include_directories(aoi_cpp_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
if(WIN32)
    target_link_libraries(aoi_cpp_bench PRIVATE psapi)
endif()
if(USE_NVML)
    target_compile_definitions(aoi_cpp_bench PRIVATE USE_NVML)
    target_include_directories(aoi_cpp_bench PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(aoi_cpp_bench PRIVATE ${NVML_LIBRARY})
endif()

# --- 2. Python extension module (the GUI bridge) -----------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Fall back to the pip-installed pybind11 in the active venv
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

pybind11_add_module(yolo_backend_cpp yolo_backend_cpp.cpp)
target_compile_definitions(yolo_backend_cpp PRIVATE BUILD_PYTHON_MODULE)
target_link_libraries(yolo_backend_cpp PRIVATE ${OpenCV_LIBS})
target_include_directories(yolo_backend_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
if(WIN32)
    target_link_libraries(yolo_backend_cpp PRIVATE psapi)
endif()
if(USE_NVML)
    target_compile_definitions(yolo_backend_cpp PRIVATE USE_NVML)
    target_include_directories(yolo_backend_cpp PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(yolo_backend_cpp PRIVATE ${NVML_LIBRARY})
endif()

# Drop the .pyd next to the Python package so `import yolo_backend_cpp` works
set_target_properties(yolo_backend_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/../aoi_gui"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../aoi_gui"
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/../aoi_gui"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../aoi_gui")

# Copy the OpenCV runtime DLL beside the .pyd (Windows loads it at import time)
if(WIN32)
    add_custom_command(TARGET yolo_backend_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DIR}/x64/vc16/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll"
                "${CMAKE_SOURCE_DIR}/../aoi_gui/"
        COMMENT "Copying OpenCV runtime next to the .pyd (ignore if path differs)"
        VERBATIM)
endif()
