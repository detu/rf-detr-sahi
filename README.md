# rf-detr-sahi

SAHI implementation in RF-DETR: Slicing Aided Hyper Inference on top of [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp).

## Layout

```
rf-detr-sahi/
├── CMakeLists.txt
├── main.cpp
├── rf-detr.cpp/              # git submodule (mudler/rf-detr.cpp)
└── models/
    └── rfdetr-medium-f16.gguf   # not committed, download separately
```

## Clone

```bash
git clone --recursive https://github.com/detu/rf-detr-sahi
# already cloned without --recursive:
git submodule update --init --recursive
```

## Model

```bash
hf download mudler/rfdetr-cpp-medium rfdetr-medium-f16.gguf --local-dir models/
```

## Build (Vulkan)

Requires CMake ≥ 3.18, a C++17 compiler, OpenCV (core, imgproc, imgcodecs, highgui, dnn),
and the Vulkan SDK (`glslc` is needed to compile ggml's Vulkan shaders).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The Vulkan backend is forced on in `CMakeLists.txt`. At runtime rf-detr.cpp uses
Vulkan device 0 and falls back to CPU if no Vulkan device is found.

## Run

```bash
./build/rfdetr_sahi models/rfdetr-medium-f16.gguf image.jpg [sahi_output.jpg]
```

Slice size is taken from the model's native input resolution (384 Nano, 512 Small,
576 Medium); tiles overlap by 20%, and detections are merged with class-aware NMS
(IoU 0.5, confidence 0.45).
