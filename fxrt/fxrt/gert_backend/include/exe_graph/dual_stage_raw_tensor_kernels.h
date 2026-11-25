#pragma once

#include <cstdint>

namespace fx_converter {

constexpr uint32_t kMaxDualStageTensorCount = 1024U;

// Converter-only layout. All logical inputs and outputs are boxed Tensor*.
// Packed RuntimeAttrs, when present, immediately follow this header.
struct DualStageTensorLayout {
  uint32_t tensor_count = 0;
  uint32_t attr_count = 0;
  uint8_t bool_attrs[2] = {};
  uint16_t reserved_ = 0;
  uint32_t output_count = 1;
  uint32_t packed_attr_bytes = 0;
  // Non-positive values preserve the caller's stream resource limits.
  int32_t cube_core_limit = 0;
  int32_t vector_core_limit = 0;
  uint32_t reserved_1_ = 0;
};

static_assert(sizeof(DualStageTensorLayout) % 8U == 0U, "trailing RuntimeAttrs must stay 8-aligned");

} // namespace fx_converter
