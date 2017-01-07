cmake_minimum_required (VERSION 3.0.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS -std=c++11 ${CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE DEBUG)
project(lab2)
find_package(OpenCL REQUIRED)

message(STATUS "OpenCL found: ${OPENCL_FOUND}")
message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
message(STATUS "OpenCL CXX includes: ${OPENCL_HAS_CPP_BINDINGS}")
message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")


include_directories(${OPENCL_INCLUDE_DIRS})

add_executable(scan main.cpp)	
target_link_libraries(scan ${OPENCL_LIBRARIES})