cmake_minimum_required(VERSION 3.6)
project(cl_face C CXX)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)
option(USE_OPENMP      "Set to ON to build use openmp"  ON)
option(USE_SSE         "Set to ON to build use SSE"  ON)

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_61,code=compute_61)

if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

# Use OpenMP
if (USE_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

#Use OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#Use Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

#Use Caffe
include_directories(/home/tpys/tools/caffe/distribute/include)
link_directories(/home/tpys/tools/caffe/distribute/lib)

#Use CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-8.0/include)
link_directories(/usr/local/cuda-8.0/lib64/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE include_files ${CMAKE_CURRENT_SOURCE_DIR}/cpp_modules/*.h)
foreach(file_path ${include_files})
    get_filename_component(dir_path ${file_path} PATH)
    include_directories(${dir_path})
endforeach()

file(GLOB_RECURSE detection_src cpp_modules/detection/src/*.cpp)
file(GLOB_RECURSE alignment_src cpp_modules/alignment/src/*.cpp cpp_modules/alignment/src/*.c)
file(GLOB_RECURSE recognition_src cpp_modules/recognition/src/*.cpp)

# Build shared library
set(BUILD_SHARED_LIBS ON)
add_library(detection_lib ${detection_src})
add_library(alignment_lib ${alignment_src})
add_library(recognition_lib ${recognition_src})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp_modules)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp_modules/openpose)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp_modules/tracking)


# Build examples
if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    list(APPEND face_app_libs detection_lib alignment_lib recognition_lib core face utilities caffe glog ${OpenCV_LIBS})
    add_executable(alignment_benchmark
            cpp_modules/example/alignment_benchmark.cpp)
    target_link_libraries(alignment_benchmark ${face_app_libs})

    add_executable(all_in_one cpp_modules/example/all_in_one.cpp
            cpp_modules/camera/src/image_warping.cpp)
    target_link_libraries(all_in_one ${face_app_libs})

    add_executable(mtcnn_alig_dataset
            cpp_modules/example/mtcnn_align_dataset.cpp)
    target_link_libraries(mtcnn_alig_dataset ${face_app_libs} boost_filesystem)
endif()
