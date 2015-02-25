cmake_minimum_required ( VERSION 2.8)

find_package(CUDA)


file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

SET (demo_files main.cpp)
    
CUDA_ADD_EXECUTABLE(executable ${demo_files} ${hdr} ${cu})


