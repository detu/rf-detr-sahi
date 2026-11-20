# rf-detr-sahi

SAHI (Slicing Aided Hyper Inference) on top of [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp).

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
