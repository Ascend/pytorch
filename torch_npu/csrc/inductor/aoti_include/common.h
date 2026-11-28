#pragma once

#include <array>
#include <filesystem>
#include <optional>

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch_npu/csrc/inductor/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
using half = at::Half;
using bfloat16 = at::BFloat16;

// Round up to the nearest multiple of 64
#if !defined(TORCH_INDUCTOR_CPP_WRAPPER) // cpp wrapper compilation takes align() from torch's header
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
#endif // !defined(TORCH_INDUCTOR_CPP_WRAPPER)
