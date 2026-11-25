#pragma once

#include <vector>

#include "aclnn_meta_api.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/types.h"

namespace fx_converter {

using fxrt::ops::aclScalar;
using fxrt::ops::aclTensor;

aclDataType ToAclDataType(ge::DataType dt);

// Runtime TensorV2 exposes non-contiguous view metadata to GE and ACLNN.
std::vector<int64_t> TensorLogicalShape(const gert::Tensor* tensor);

// True when stride is a last-two-dim swap of a contiguous logical layout
// (ATen t() / transpose(-2,-1)). TensorV1 is dense.
bool IsTensorLastTwoTransposed(const gert::Tensor* tensor);

// Build a contiguous ND aclTensor view over gert::Tensor device memory.
aclTensor* ToAclTensor(const gert::Tensor* tensor);

// Same as ToAclTensor but override the dtype (e.g. gather output must match input dtype).
aclTensor* ToAclTensorWithDtype(const gert::Tensor* tensor, ge::DataType dtype);

// Last-two transpose view: view shape/stride swap the trailing dims, storage stays physical.
aclTensor* ToAclTensorTransposedLastTwo(const gert::Tensor* tensor);

// Reconstruct a transposed logical feed whose stride was dropped by an ES dataset.
aclTensor* ToAclTensorAssumingLastTwoTransposed(const gert::Tensor* tensor);

// Expand a tensor as a stride-0 ACL view. TensorV2 broadcasts from its current
// logical size/stride; invalid view broadcasts fail instead of silently densifying.
aclTensor* ToAclTensorBroadcastTo(const gert::Tensor* tensor, const std::vector<int64_t>& logical_shape);

void DestroyAclTensor(aclTensor* t);
void DestroyAclScalar(aclScalar* s);

} // namespace fx_converter
