cmake_minimum_required(VERSION 3.5)
project(cl_face C CXX)

# Build options
option(BUILD_EXAMPLES  "Set to ON to build examples"  ON)
option(USE_OPENMP      "Set to ON to build use openmp"  ON)
option(USE_SSE         "Set to ON to build use SSE"  ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# Use SSE
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/detection/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/alignment/include)

file(GLOB detect_src
        detection/src/util/*.cpp
        detection/src/io/*.cpp
        detection/src/feat/*.cpp
        detection/src/classifier/*.cpp
        detection/src/*.cpp)

file(GLOB alignment_src alignment/src/*.cpp)


# Build shared library
add_library(detect_lib SHARED ${detect_src})
add_library(alignment_lib SHARED ${alignment_src})
set(detect_required_libs detect_lib)
set(alignment_required_libs alignment_lib)


# Build examples
if (BUILD_EXAMPLES)
    message(STATUS "Build with examples.")
    find_package(OpenCV)
    if (NOT OpenCV_FOUND)
        message(WARNING "OpenCV not found. Test will not be built.")
    else()
        include_directories(${OpenCV_INCLUDE_DIRS})
        list(APPEND face_app_libs ${OpenCV_LIBS} detect_lib alignment_lib)
        add_executable(face_app example/face_app_example.cpp)
        target_link_libraries(face_app ${face_app_libs})
    endif()
endif()
