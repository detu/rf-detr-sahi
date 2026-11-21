# rf-detr-sahi

Tiled RF-DETR inference on top of [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp): the
image is cut into model-sized tiles overlapping by 50%, each tile is run through RF-DETR at
native resolution, detections cut by an interior tile edge are dropped (the neighbouring
tile sees those objects whole), and the rest are merged with one NMS. Small objects in
large images stay at native resolution.

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
| `-o, --output` | `output.jpg` / `output/` | output file for one image, output directory for several (`<name>_det.<ext>`) |
| `-n, --names` | model's names | text file of class names, one per line: line 1 = class 0, line 2 = class 1, ... |
| `--csv` | off | write all detections: `image,class_id,class_name,score,x1,y1,x2,y2` |
| `-r, --recursive` | off | scan directories recursively |
| `-w, --workers` | 2 | tiles processed in parallel (OpenMP); each worker has its own RF-DETR context |
| `-t, --threads` | cores / workers | ggml CPU threads per worker |
| `--stride` | half a tile | tile stride in pixels (default: 50% overlap) |
| `--margin` | 8 | drop detections within this many pixels of an interior tile edge |
| `--pre-scale` | 1 | downscale the image by this factor (0–1] before tiling: fewer tiles, larger objects fit |
| `--conf` | 0.45 | confidence threshold |
| `--iou` | 0.50 | NMS IoU threshold for merging the tiles |

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

### How the tiling works

1. **Work image.** With `--pre-scale f < 1` the image is first downscaled (area
   averaging). If it is smaller than one tile, it is padded at the bottom/right with gray
   (114) to one tile, so coordinates don't move.
2. **Tile grid.** Tiles are the model's native input size (384 Nano, 512 Small, 576
   Medium), so every tile reaches RF-DETR square and unscaled. They are placed every
   `stride` pixels (default half a tile, i.e. 50% overlap); the number of tiles per
   row/column is the smallest that covers the image, and the last tile is clamped to the
   image border, so there are no partial or duplicate tiles.
3. **Edge filter.** A detection within `--margin` pixels of an *interior* tile edge (an
   edge that isn't the image border) is dropped: it is probably cut, and with 50% overlap
   the neighbouring tile sees the object whole. Detections on the real image border are
   kept.
4. **Merge.** The rest is mapped back to the original image (undoing the pre-scale),
   clipped to the real image (dropping the gray padding) and merged with one class-aware
   NMS, which removes objects seen whole by two overlapping tiles.

Output is deterministic regardless of the number of workers.

Objects larger than about a tile can't be seen whole by any tile, so the edge filter
drops them. Use `--pre-scale` to shrink such images until the large objects fit in a
tile. With 50% overlap there are about 4× as many tiles as without overlap, so
`--pre-scale` is also the main speed control.

Each worker loads its own copy of the model (about 64 MB for Medium F16, in GPU
memory with CUDA/Vulkan). With a single GPU, 1–2 workers and 2–4 threads each is
usually best; on CPU, keep `workers × threads` at or below your physical core count.
