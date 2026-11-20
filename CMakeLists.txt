cmake_minimum_required(VERSION 3.18)
project(rfdetr_sahi CXX)

# Default to Release: a Debug build makes ggml many times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compute backend for rf-detr.cpp: CUDA (default), VULKAN or CPU.
#   -DSAHI_BACKEND=CUDA    NVIDIA GPU; needs the CUDA toolkit (nvcc). Fast path under WSL2.
#                          ggml builds only for the GPU in this machine when it can
#                          (CMake >= 3.24, CUDA >= 11.6); override with -DCMAKE_CUDA_ARCHITECTURES=89.
#   -DSAHI_BACKEND=VULKAN  native Vulkan driver (very slow under WSL2's Dozen driver)
#   -DSAHI_BACKEND=CPU     CPU only
set(SAHI_BACKEND "CUDA" CACHE STRING "Compute backend: CUDA, VULKAN or CPU")
set_property(CACHE SAHI_BACKEND PROPERTY STRINGS CUDA VULKAN CPU)
string(TOUPPER "${SAHI_BACKEND}" SAHI_BACKEND)

set(RFDETR_GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(RFDETR_GGML_CUDA   OFF CACHE BOOL "" FORCE)
if(SAHI_BACKEND STREQUAL "VULKAN")
    set(RFDETR_GGML_VULKAN ON CACHE BOOL "" FORCE)
elseif(SAHI_BACKEND STREQUAL "CUDA")
    set(RFDETR_GGML_CUDA ON CACHE BOOL "" FORCE)
elseif(NOT SAHI_BACKEND STREQUAL "CPU")
    message(FATAL_ERROR "SAHI_BACKEND must be CUDA, VULKAN or CPU (got '${SAHI_BACKEND}')")
endif()
message(STATUS "rfdetr_sahi backend: ${SAHI_BACKEND} (${CMAKE_BUILD_TYPE})")

set(RFDETR_BUILD_CLI OFF CACHE BOOL "" FORCE)

add_subdirectory(rf-detr.cpp)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(rfdetr_sahi main.cpp)

target_include_directories(rfdetr_sahi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rf-detr.cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rfdetr_sahi PRIVATE
    rfdetr  # Target name exported by rf-detr.cpp/CMakeLists.txt
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)
