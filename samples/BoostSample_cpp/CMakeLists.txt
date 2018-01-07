cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra")
add_executable(BoostSample main.cpp)

find_package(Boost REQUIRED COMPONENTS filesystem system)
if (Boost_FOUND)
  message(STATUS "Boost library directory = ${Boost_LIBRARY_DIRS}")
  message(STATUS "Boost include directory = ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
endif ()

target_link_libraries(BoostSample ${Boost_LIBRARIES})
