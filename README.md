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

## Build

Requires CMake ≥ 3.18, a C++17 compiler with OpenMP, and OpenCV (core, imgproc,
imgcodecs, highgui, dnn). The backend defaults to **CUDA** and the build type to
**Release**:

```bash
cmake -B build                         # CUDA (needs the CUDA toolkit / nvcc)
cmake --build build -j 2               # CUDA files need 1-3 GB RAM each; raise -j if you have the memory
```

Other backends:

```bash
cmake -B build -DSAHI_BACKEND=VULKAN   # needs Vulkan SDK, SPIRV-Headers, glslc
cmake -B build -DSAHI_BACKEND=CPU
```

ggml builds CUDA code only for the GPU in this machine when it can (CMake ≥ 3.24,
CUDA ≥ 11.6); otherwise pass e.g. `-DCMAKE_CUDA_ARCHITECTURES=89` (Ada). If CMake
can't find nvcc, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. At runtime the
GPU backends use device 0 and fall back to CPU if none is found.

**WSL2:** Vulkan in WSL2 goes through Mesa's Dozen driver (Vulkan on Direct3D 12),
shown as `Microsoft Direct3D12 (...) (Dozen)` at startup, and is very slow for ggml.
On an NVIDIA GPU use CUDA (the default): it runs natively in WSL2. Install only the
toolkit (`cuda-toolkit` from NVIDIA's `wsl-ubuntu` repo), never a Linux GPU driver.
If the build is `Killed` (error 137), WSL ran out of memory: use `-j 1`/`-j 2` or give
WSL more memory/swap in `%UserProfile%\.wslconfig`.

## Run

The model is loaded once; every image given is processed with it. Inputs can be
image files and/or directories.

```bash
# one image
./build/rfdetr_sahi models/rfdetr-medium-f16.gguf image.jpg -o result.jpg

# a whole folder (add -r for subfolders), with a CSV of all detections
./build/rfdetr_sahi models/rfdetr-medium-f16.gguf images/ -o results/ --csv detections.csv
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `sahi_output.jpg` / `sahi_output/` | output file for one image, output directory for several (`<name>_sahi.<ext>`) |
| `-n, --names` | model's names | text file of class names, one per line: line 1 = class 0, line 2 = class 1, ... |
| `--csv` | off | write all detections: `image,class_id,class_name,score,x1,y1,x2,y2` |
| `-r, --recursive` | off | scan directories recursively |
| `-w, --workers` | 2 | tiles processed in parallel (OpenMP); each worker has its own RF-DETR context |
| `-t, --threads` | cores / workers | ggml CPU threads per worker |
| `--overlap` | 0.20 | tile overlap ratio |
| `--conf` | 0.45 | confidence threshold |
| `--iou` | 0.50 | NMS IoU threshold |

With `--names`, the labels on the output images and the `class_name` column in the
CSV use the names from the file. Windows line endings and surrounding spaces are
ignored. An empty line, or a class id beyond the end of the file, keeps the model's
own name for that class.

```bash
./build/rfdetr_sahi models/rfdetr-medium-f16.gguf images/ -o results/ -n classes.txt
```

Directories are scanned for `.jpg .jpeg .png .bmp .tif .tiff .webp` (sorted by name).
The next image is decoded in the background while the current one is processed.
Images that fail to load are skipped and reported; the exit code is non-zero if any failed.

Slice size is taken from the model's native input resolution (384 Nano, 512 Small,
576 Medium). Detections are merged with class-aware NMS. Output is deterministic
regardless of the number of workers.

Each worker loads its own copy of the model (about 64 MB for Medium F16, in GPU
memory with CUDA/Vulkan). With a single GPU, 1–2 workers and 2–4 threads each is
usually best; on CPU, keep `workers × threads` at or below your physical core count.
