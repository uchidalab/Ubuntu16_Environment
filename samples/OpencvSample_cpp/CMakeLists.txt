cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra")
add_executable(OpencvSample main.cpp)

find_package(OpenCV REQUIRED COMPONENTS)
if (OPENCV_FOUND)
  message(STATUS "OpenCV library directory = ${OpenCV_LIBRARY_DIRS}")
  message(STATUS "OpenCV include directory = ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

target_link_libraries(OpencvSample ${OpenCV_LIBS})
