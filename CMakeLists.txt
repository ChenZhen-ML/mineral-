cmake_minimum_required(VERSION 3.21)
project(task)

set(CMAKE_CXX_STANDARD 14)

add_executable(task main.cpp)

set(OpenCV_DIR D:\\OPENCV.COMPLIER)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(task ${OpenCV_LIBS})