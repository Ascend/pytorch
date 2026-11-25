#!/usr/bin/env bash
set -euo pipefail

python=$(command -v "${1:-python3}")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core=$(cd "${2:-$root/libtorch_npu}" && pwd)

run_mixed_wheel() (
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    src="$root/examples/libtorch_mixed_wheel"
    build="$src/build"
    torch_cmake=$("$python" -I -c 'import torch; print(torch.utils.cmake_prefix_path)')
    site=$("$python" -I -c 'import sysconfig; print(sysconfig.get_path("platlib"))')

    cmake -S "$src" -B "$build" \
        -DLIBTORCH_NPU_ROOT="$core" \
        -DCMAKE_PREFIX_PATH="$torch_cmake" \
        -DPython3_EXECUTABLE="$python"
    cmake --build "$build"

    if PYTHONPATH="$site" LD_LIBRARY_PATH="$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$build/mixed_wheel" "$core/lib/libtorch_npu.so"; then
        echo "Mixed wheel: Success."
    else
        echo "Mixed wheel: Failed." >&2
        exit 1
    fi
)

bash "$root/ci/libtorch_resnet.sh" "${1:-python3}"
run_mixed_wheel
