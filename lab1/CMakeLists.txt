cmake_minimum_required (VERSION 3.0.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(lab1)
find_package( OpenCL REQUIRED )

message(STATUS "OpenCL found: ${OPENCL_FOUND}")
message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
message(STATUS "OpenCL CXX includes: ${OPENCL_HAS_CPP_BINDINGS}")
message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")


include_directories( ${OPENCL_INCLUDE_DIRS} )

add_executable(convolution main.cpp)	
# target_include_directories( convolution ${OPENCL_INCLUDE_DIR} )
target_link_libraries( convolution ${OPENCL_LIBRARIES} )