/* Copyright 2026 Huawei Technologies Co., Ltd */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exe_graph/runtime/tensor.h" // GE_HEADER
#include "shape/fx_fused_infer_shape.h"

namespace fx_converter {

constexpr uint32_t kFxConstPayloadMagic = 0x1234ABCDU;
constexpr uint32_t kFxInferFuncAbiVersion = 1U;

enum class FxConstPayloadType : uint32_t {
  kInferFunc = 1U,
};

// The object is process-local, but its fields are deliberately POD so the OM2
// generic serialized-buffer restorer can rebuild the same view without knowing
// the payload format.
struct FxSerializedBuffer {
  const uint8_t* data = nullptr;
  uint64_t size = 0U;
};

using FxInferShapeFunc =
    int32_t (*)(const gert::Tensor* const* inputs, size_t input_num, gert::Tensor* const* outputs, size_t output_num);

struct FxInferFuncHandle {
  void* so_handle = nullptr;
  FxInferShapeFunc func = nullptr;
};

struct FxInferFuncArtifact {
  std::string so_path;
  std::string symbol;
};

bool BuildInferShapeSharedObject(const FxFusedShapeRules& rules, FxInferFuncArtifact* artifact);
bool SerializeInferFuncPayload(const FxInferFuncArtifact& artifact, std::vector<uint8_t>* payload);
bool ParseInferFuncPayload(const FxSerializedBuffer& buffer, FxInferFuncArtifact* artifact);
ge::graphStatus DeserializeInferFunc(const FxSerializedBuffer& buffer, FxInferFuncHandle** handle);

} // namespace fx_converter
