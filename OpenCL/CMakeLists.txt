cmake_minimum_required ( VERSION 2.8)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_module")
find_package( OpenCL REQUIRED )
include_directories( ${OPENCL_INCLUDE_DIRS} )

add_executable( executable main.cpp )
target_link_libraries( executable ${OPENCL_LIBRARIES} )

configure_file(kernels.cl ${CMAKE_CURRENT_BINARY_DIR}/kernels.cl COPYONLY)
