cmake_minimum_required(VERSION 3.18)
project(multiple_devices_single_process LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# NCCL: prefer NCCL_HOME if set, otherwise search system paths
if(DEFINED ENV{NCCL_HOME})
    set(NCCL_HOME $ENV{NCCL_HOME})
endif()

if(NCCL_HOME)
    find_path(NCCL_INCLUDE_DIR nccl.h PATHS ${NCCL_HOME}/include NO_DEFAULT_PATH)
    find_library(NCCL_LIBRARY nccl PATHS ${NCCL_HOME}/lib NO_DEFAULT_PATH)
else()
    find_path(NCCL_INCLUDE_DIR nccl.h)
    find_library(NCCL_LIBRARY nccl)
endif()

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR
        "NCCL not found. Set NCCL_HOME to the NCCL install prefix, e.g.:\n"
        "  cmake -DNCCL_HOME=/path/to/nccl/build ..")
endif()

message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")

add_executable(multiple_devices_single_process main.cc)

target_include_directories(multiple_devices_single_process PRIVATE
    ${NCCL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(multiple_devices_single_process PRIVATE
    ${NCCL_LIBRARY}
    CUDA::cudart
)

# Set RPATH so the binary can find libnccl.so at runtime
get_filename_component(NCCL_LIB_DIR ${NCCL_LIBRARY} DIRECTORY)
set_target_properties(multiple_devices_single_process PROPERTIES
    BUILD_RPATH ${NCCL_LIB_DIR}
    INSTALL_RPATH ${NCCL_LIB_DIR}
)
