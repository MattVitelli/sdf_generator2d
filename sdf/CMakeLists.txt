cmake_minimum_required(VERSION 3.6)

project(sdf)

include(cmake_utils)

add_executable(sdf "")

add_src(src/main.cpp)

find_package(OpenCV REQUIRED HINTS ${CMAKE_MODULE_PATH})

target_include_directories(sdf PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(sdf PRIVATE ${OpenCV_LIBS})
target_compile_definitions(sdf PUBLIC OPENCV_VERSION_3)
