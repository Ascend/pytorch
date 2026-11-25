// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FXRT_INCLUDE_TORCH_FALLBACK_ABI_H_
#define FXRT_INCLUDE_TORCH_FALLBACK_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FX_TORCH_FALLBACK_ABI_VERSION 3U

// Rank caps shared by both sides of the ABI; the only valid storage_stride /
// stride_capacity values a caller may pass to export_layout / export_strides.
// run_deferred rejects a result whose logical rank exceeds
// FX_TORCH_FALLBACK_MAX_RANK (the metadata sizes array below is the limiting
// factor). Physical storage ranks of private formats are much smaller than
// logical ranks, so export_layout rows are capped by the storage constant.
#define FX_TORCH_FALLBACK_MAX_RANK 32U
#define FX_TORCH_FALLBACK_MAX_STORAGE_RANK 8U

typedef enum FxTorchValueKind {
  FX_TORCH_VALUE_NONE = 0,
  FX_TORCH_VALUE_TENSOR = 1,
  FX_TORCH_VALUE_DOUBLE = 2,
  FX_TORCH_VALUE_INT = 3,
  FX_TORCH_VALUE_BOOL = 4,
  FX_TORCH_VALUE_STRING = 5,
  FX_TORCH_VALUE_LIST = 6,
} FxTorchValueKind;

typedef struct FxTorchStringView {
  const char* data;
  size_t size;
} FxTorchStringView;

typedef struct FxTorchTensorView {
  void* data;
  const int64_t* sizes;
  const int64_t* strides;
  size_t rank;
  int64_t storage_offset;
  int32_t dtype;
  // NPU private storage format id (aclFormat, e.g. FRACTAL_NZ=29). 0 or ND
  // means a plain dense tensor; any other value tags the (already laid out)
  // buffer with that private format on the Torch side. reserved keeps the
  // pre-refactor field name at the same offset.
  union {
    uint32_t reserved;
    uint32_t storage_format;
  };
  size_t nbytes;
  const int64_t* storage_sizes;
  size_t storage_rank;
} FxTorchTensorView;

typedef struct FxTorchValue {
  uint32_t kind;
  uint32_t reserved;
  FxTorchTensorView tensor;
  int64_t int_value;
  double double_value;
  FxTorchStringView string_value;
  const struct FxTorchValue* elements;
  size_t element_count;
} FxTorchValue;

typedef struct FxTorchArgSpec {
  uint32_t kind;
  uint32_t element_kind;
} FxTorchArgSpec;

typedef struct FxTorchOpSpec {
  FxTorchStringView qualified_name;
  FxTorchStringView overload_name;
  const FxTorchArgSpec* arguments;
  size_t argument_count;
  size_t output_count;
} FxTorchOpSpec;

typedef struct FxTorchError {
  char* message;
  size_t capacity;
} FxTorchError;

typedef struct FxTorchFallbackResultMetadata {
  uint32_t rank;
  int32_t dtype;
  int64_t sizes[FX_TORCH_FALLBACK_MAX_RANK];
} FxTorchFallbackResultMetadata;

typedef struct FxTorchFallbackApi {
  uint32_t abi_version;
  uint32_t struct_size;
  void* (*create_plan)(const FxTorchOpSpec* spec, FxTorchError* error);
  int32_t (*run_into)(
      void* plan,
      const FxTorchValue* arguments,
      size_t argument_count,
      const FxTorchTensorView* outputs,
      size_t output_count,
      uintptr_t stream,
      int32_t device_id,
      FxTorchError* error);
  int32_t (*run_deferred)(
      void* plan,
      const FxTorchValue* arguments,
      size_t argument_count,
      uintptr_t stream,
      int32_t device_id,
      uintptr_t* result_handle,
      FxTorchFallbackResultMetadata* metadata,
      size_t metadata_count,
      FxTorchError* error);
  int32_t (*materialize)(
      uintptr_t result_handle,
      const FxTorchTensorView* outputs,
      size_t output_count,
      uintptr_t stream,
      int32_t device_id,
      FxTorchError* error);
  void (*destroy_result)(uintptr_t result_handle);
  void (*destroy_plan)(void* plan);
  // 1 when create_plan selected an out= schema. Functional plans must use
  // ExecuteCustomOp + run_deferred so a stale InferShape template cannot pin
  // GE's output allocation; WithInferShape skips realloc when size > 0.
  int32_t (*uses_out_schema)(void* plan);
  // Schema-derived ref pairs for mutating (out-param) ops on non-out
  // overloads: (schema argument index of a write-alias Tensor arg, output
  // index of the aliased return). The converter maps the schema arg index to
  // the eager tensor-input slot through the call's argument tree and calls
  // EagerOpExecutionContext::MakeOutputRefInput(output, slot) so GE shares
  // the input buffer with the output. May be null on older bridges.
  size_t (*ref_pairs)(void* plan, size_t* schema_arg_indices, size_t* output_indices, size_t capacity);
  // Publish the already-allocated Torch storages from run_deferred. The
  // pointers stay valid until destroy_result. Used so GE can adopt them
  // instead of a second L1 malloc + D2D copy (that malloc is what surfaces
  // as "failed to allocate deferred output" once HBM is tight).
  int32_t (*export_storage)(uintptr_t result_handle, void** addrs, size_t* nbytes, size_t count, FxTorchError* error);
  // aclFormat and physical storage sizes of each deferred result. storage_sizes
  // is row-major with storage_stride columns. Pointers stay valid until
  // destroy_result. May be null on an older bridge; the caller then keeps ND.
  int32_t (*export_layout)(
      uintptr_t result_handle,
      int32_t* formats,
      int64_t* storage_ranks,
      int64_t* storage_sizes,
      size_t storage_stride,
      size_t count,
      FxTorchError* error);
  // Logical strides of each deferred result. flat_strides is row-major with
  // stride_capacity columns. offsets[i] is the storage offset. May be null on
  // an older bridge; the caller then keeps a contiguous view.
  int32_t (*export_strides)(
      uintptr_t result_handle,
      int64_t* ranks,
      int64_t* flat_strides,
      int64_t* offsets,
      size_t stride_capacity,
      size_t count,
      FxTorchError* error);
} FxTorchFallbackApi;

#ifdef __cplusplus
}
#endif

#endif // FXRT_INCLUDE_TORCH_FALLBACK_ABI_H_
