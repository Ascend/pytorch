#pragma once

#include <torch/version.h>

// Returns 1 if the current PyTorch version is >= MAJOR.MINOR, 0 otherwise.
// Use in #if expressions to gate version-specific code.
//
// COMPAT blocks should be annotated as:
//   // COMPAT(>= X.Y): <description of upstream change>
//   // CAN REMOVE <branch> when MIN_SUPPORTED >= (X, Y)
//
// Bump MIN_SUPPORTED when dropping old version support.
#define TORCH_NPU_VERSION_GE(MAJOR, MINOR) \
    (TORCH_VERSION_MAJOR > (MAJOR) || \
     (TORCH_VERSION_MAJOR == (MAJOR) && TORCH_VERSION_MINOR >= (MINOR)))

// Keep in sync with MIN_SUPPORTED_VERSION in torch_npu/_compat/version.py.
#define TORCH_NPU_MIN_SUPPORTED_MAJOR 2
#define TORCH_NPU_MIN_SUPPORTED_MINOR 10
