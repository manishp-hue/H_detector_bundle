cmake_minimum_required(VERSION 3.16)
project(newspace_air_fresh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui video videoio imgcodecs)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0>=1.20
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-rtsp-server-1.0
    gstreamer-rtsp-1.0
    gio-2.0)

add_executable(publisher_node
    src/publisher_node.cpp
    tracker/src/vision.cpp
    tracker/src/tracker.cpp
    tracker/src/trt_runtime.cpp)

target_include_directories(publisher_node PRIVATE
    tracker/include
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(publisher_node PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    PkgConfig::GST)

target_compile_options(publisher_node PRIVATE -O3 -Wall -Wextra -Wpedantic)

option(ENABLE_TENSORRT "Enable TensorRT AI recovery" ON)
if(ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include /usr/include/aarch64-linux-gnu /usr/local/cuda/include)
    find_library(TENSORRT_LIB nvinfer
        PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib)
    find_library(CUDART_LIB cudart
        PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu /usr/lib)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB AND CUDART_LIB)
        message(STATUS "TensorRT enabled: ${TENSORRT_LIB}")
        target_compile_definitions(publisher_node PRIVATE HELIPAD_USE_TENSORRT=1)
        target_include_directories(publisher_node PRIVATE
            ${TENSORRT_INCLUDE_DIR}
            /usr/local/cuda/include)
        target_link_libraries(publisher_node PRIVATE
            ${TENSORRT_LIB}
            ${CUDART_LIB})
    else()
        message(WARNING "TensorRT not found; geometry-only runtime will build")
    endif()
endif()
