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
./build/rfdetr_sahi models/rfdetr-medium-f16.gguf image.jpg -o sahi_output.jpg -w 2 -t 8
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `sahi_output.jpg` | output image |
| `-w, --workers` | 2 | tiles processed in parallel (OpenMP); each worker has its own RF-DETR context |
| `-t, --threads` | cores / workers | ggml CPU threads per worker |
| `--overlap` | 0.20 | tile overlap ratio |
| `--conf` | 0.45 | confidence threshold |
| `--iou` | 0.50 | NMS IoU threshold |

Slice size is taken from the model's native input resolution (384 Nano, 512 Small,
576 Medium). Detections are merged with class-aware NMS, with classes processed in
parallel. Output is deterministic regardless of the number of workers.

Each worker loads its own copy of the model (about 64 MB for Medium F16, in VRAM
with Vulkan). On CPU, keep `workers × threads` at or below your core count. With
Vulkan the GPU runs the tiles largely one after another, so 2 workers (overlapping
one tile's CPU work with another's GPU work) is usually the sweet spot.
