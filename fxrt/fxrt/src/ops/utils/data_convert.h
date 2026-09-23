#ifndef FXRT_FXRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_
#define FXRT_FXRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_

#include <torch/extension.h>
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "common/visible.h"
namespace fxrt::ops {

FXRT_EXPORT at::Tensor ToTorchTensor(const ir::TensorPtr& tensor);
FXRT_EXPORT ir::TensorPtr FromTorchTensor(const at::Tensor& tensor, bool isFake = false);
FXRT_EXPORT void UpdateTensorFromTorch(const ir::TensorPtr& tensor, const at::Tensor& atTensor);
FXRT_EXPORT void CheckOutputInputRef(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    const std::string& opName);
FXRT_EXPORT bool IsTorchTensorStandardLayout(const at::Tensor& tensor);
} // namespace fxrt::ops
#endif // FXRT_FXRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_
