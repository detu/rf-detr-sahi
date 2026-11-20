cmake_minimum_required(VERSION 3.18)
project(rfdetr_sahi_vulkan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Vulkan backend inside rf-detr.cpp
set(RFDETR_GGML_VULKAN ON CACHE BOOL "" FORCE)
set(RFDETR_BUILD_CLI OFF CACHE BOOL "" FORCE)

add_subdirectory(rf-detr.cpp)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)

add_executable(rfdetr_sahi main.cpp)

target_include_directories(rfdetr_sahi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rf-detr.cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rfdetr_sahi PRIVATE
    rfdetr  # Target name exported by rf-detr.cpp/CMakeLists.txt
    ${OpenCV_LIBS}
)
