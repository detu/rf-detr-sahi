cmake_minimum_required(VERSION 3.18)
project(rfdetr_sahi_vulkan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compute backend for rf-detr.cpp: VULKAN, CUDA or CPU.
#   -DSAHI_BACKEND=CUDA    NVIDIA GPU (also the fast path under WSL2)
#   -DSAHI_BACKEND=VULKAN  native Vulkan driver (slow under WSL2's Dozen driver)
#   -DSAHI_BACKEND=CPU     CPU only
set(SAHI_BACKEND "VULKAN" CACHE STRING "Compute backend: VULKAN, CUDA or CPU")
set_property(CACHE SAHI_BACKEND PROPERTY STRINGS VULKAN CUDA CPU)
string(TOUPPER "${SAHI_BACKEND}" SAHI_BACKEND)

set(RFDETR_GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(RFDETR_GGML_CUDA   OFF CACHE BOOL "" FORCE)
if(SAHI_BACKEND STREQUAL "VULKAN")
    set(RFDETR_GGML_VULKAN ON CACHE BOOL "" FORCE)
elseif(SAHI_BACKEND STREQUAL "CUDA")
    set(RFDETR_GGML_CUDA ON CACHE BOOL "" FORCE)
elseif(NOT SAHI_BACKEND STREQUAL "CPU")
    message(FATAL_ERROR "SAHI_BACKEND must be VULKAN, CUDA or CPU (got '${SAHI_BACKEND}')")
endif()
message(STATUS "rfdetr_sahi backend: ${SAHI_BACKEND}")

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
