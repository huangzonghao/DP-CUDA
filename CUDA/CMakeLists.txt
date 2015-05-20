cmake_minimum_required ( VERSION 2.8)

find_package(CUDA)


file( GLOB_RECURSE  HDR  *.hpp *.h )
file( GLOB_RECURSE  CU  *.cu)
SET (CPP main.cpp)
    
CUDA_ADD_EXECUTABLE(executable ${CPP} ${HDR} ${CU})


