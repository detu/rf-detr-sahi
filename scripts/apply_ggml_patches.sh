#!/usr/bin/env bash
# Forwarder for rf-detr.cpp's ggml patch script.
#
# rf-detr.cpp's CMakeLists.txt calls ${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh.
# When rf-detr.cpp is pulled in with add_subdirectory(), CMAKE_SOURCE_DIR is this
# repo's root, so CMake looks for the script here. Forward to the real one.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../rf-detr.cpp/scripts/apply_ggml_patches.sh" "$@"
