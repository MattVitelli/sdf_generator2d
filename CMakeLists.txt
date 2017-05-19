cmake_minimum_required(VERSION 3.6)

project(sdfmaker LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(MSVC)
	list(APPEND CMAKE_MODULE_PATH "C:/src/third_party_prebuilt/opencv/install")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCV REQUIRED HINTS ${CMAKE_MODULE_PATH})

add_subdirectory(sdf)
