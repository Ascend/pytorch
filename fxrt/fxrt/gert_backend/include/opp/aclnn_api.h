#pragma once

#include <string>

#include "aclnn_op_api.h"
#include "aclnn_symbol_api.h"
#include "aclnn_meta_api.h"

namespace fx_converter {

using fxrt::ops::aclIntArray;
using fxrt::ops::aclnnStatus;
using fxrt::ops::aclOpExecutor;
using fxrt::ops::aclScalar;
using fxrt::ops::aclTensor;
using fxrt::ops::aclTensorList;
using fxrt::ops::kAclnnSuccess;

namespace aclnn_api {

// Populated during graph lowering, before Prepare/Launch can execute. The
// existing OPP binding mutex serializes this function; it is never a hot-path API.
// Unused operators do not require their CANN symbols to be present.
bool EnsureForOp(const char* opp_type, std::string* error);

// Addresses are resolved during graph construction. Prepare/Launch only deduce
// the pointer type and call it; no resolver, lock or cache lookup runs here.
// Use exact ABI types, including typed null pointers and integer widths.
template <typename... Args>
aclnnStatus Call(void* address, Args... args) {
  static_assert(
      (!std::is_same_v<std::decay_t<Args>, std::nullptr_t> && ...),
      "ACLNN optional arguments require typed null pointers");
  auto params = std::make_tuple(args...);
  auto function = fxrt::ops::ConvertToOpApiFunc(params, address);
  return fxrt::ops::CallOpApiFunc(function, params);
}

extern fxrt::ops::_aclCreateTensorFuncPtr aclCreateTensor;
extern fxrt::ops::_aclCreateScalarFuncPtr aclCreateScalar;
extern fxrt::ops::_aclCreateIntArrayFuncPtr aclCreateIntArray;
extern fxrt::ops::_aclCreateTensorListFuncPtr aclCreateTensorList;
extern fxrt::ops::_aclDestroyTensorFuncPtr aclDestroyTensor;
extern fxrt::ops::_aclDestroyScalarFuncPtr aclDestroyScalar;
extern fxrt::ops::_aclDestroyIntArrayFuncPtr aclDestroyIntArray;
extern fxrt::ops::_aclDestroyTensorListFuncPtr aclDestroyTensorList;
extern void* aclnnAddGetWorkspaceSize;
extern void* aclnnAdd;
extern void* aclnnAddmmGetWorkspaceSize;
extern void* aclnnAddmm;
extern void* aclnnAmaxGetWorkspaceSize;
extern void* aclnnAmax;
extern void* aclnnBaddbmmGetWorkspaceSize;
extern void* aclnnBaddbmm;
extern void* aclnnBatchMatMulGetWorkspaceSize;
extern void* aclnnBatchMatMul;
extern void* aclnnBitwiseNotGetWorkspaceSize;
extern void* aclnnBitwiseNot;
extern void* aclnnCatGetWorkspaceSize;
extern void* aclnnCat;
extern void* aclnnConvolutionGetWorkspaceSize;
extern void* aclnnConvolution;
extern void* aclnnFlashAttentionScoreGetWorkspaceSize;
extern void* aclnnFlashAttentionScore;
extern void* aclnnGatherGetWorkspaceSize;
extern void* aclnnGather;
extern void* aclnnIndexGetWorkspaceSize;
extern void* aclnnIndex;
extern void* aclnnInplaceCopyGetWorkspaceSize;
extern void* aclnnInplaceCopy;
extern void* aclnnInplaceFillScalarGetWorkspaceSize;
extern void* aclnnInplaceFillScalar;
extern void* aclnnMatmulGetWorkspaceSize;
extern void* aclnnMatmul;
extern void* aclnnMmGetWorkspaceSize;
extern void* aclnnMm;
extern void* aclnnReduceSumGetWorkspaceSize;
extern void* aclnnReduceSum;
extern void* aclnnSWhereGetWorkspaceSize;
extern void* aclnnSWhere;

} // namespace aclnn_api
} // namespace fx_converter
