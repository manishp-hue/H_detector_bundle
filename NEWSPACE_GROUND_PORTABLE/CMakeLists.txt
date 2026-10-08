cmake_minimum_required(VERSION 3.16)
project(newspace_ground_fresh LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets Network)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0>=1.20
    gstreamer-app-1.0
    gstreamer-video-1.0)
add_executable(subscriber_node src/subscriber_node.cpp)
target_link_libraries(subscriber_node PRIVATE
    Qt6::Widgets Qt6::Network Threads::Threads PkgConfig::GST)
target_compile_options(subscriber_node PRIVATE -O2 -Wall -Wextra -Wpedantic)
