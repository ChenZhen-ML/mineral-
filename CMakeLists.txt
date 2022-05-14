cmake_minimum_required( VERSION 2.8 )
project( Mine_direction )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
set(OpenCV_DIR /home/wjy/OPENCV/MINGW_BUILD/install)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package( OpenCV 4 REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
message("OPENCV INC DIR:${OpenCV_INCLUDE_DIRS}")
add_executable( Mine_direction main.cpp )

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# 链接OpenCV库
target_link_libraries( Mine_direction ${OpenCV_LIBS} )
