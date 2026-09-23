#include <vector>

#include "ops/ascend/mem/memcpy_likes.h"
#include "ops/utils/utils.h"
#include "ops/op_register.h"
#include "acl/acl_rt.h"
#include "ir/tensor/tensor.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace fxrt {
namespace ops {
OpsErrorCode MemcpyOpBase::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const auto& inputTensor = input[kIndex0]->ToTensor();
  const auto& outTensor = output->ToTensor();
  if (!inputTensor->IsContiguous() || inputTensor->StorageOffset() != 0 || !IsTensorBaseFormat(inputTensor) ||
      !IsTensorBaseFormat(outTensor)) {
    RT_GLOG(EXCEPTION) << "memcpy_likes operator does not support non-standard tensor memory layout, "
                       << "but got strides: " << inputTensor->Strides() << ", offset: " << inputTensor->StorageOffset()
                       << ", inputTensor format: " << FormatEnumToStr(inputTensor->Format())
                       << ", outTensor format: " << FormatEnumToStr(outTensor->Format());
  }

  return SUCCESS;
}

OpsErrorCode MemcpyOpBase::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool MemcpyOpBase::NeedLaunch() {
  return false;
}

FXRT_REG_OP(unsqueeze, Unsqueeze, Ascend);
FXRT_REG_OP(squeeze, Squeeze, Ascend);
} // namespace ops
} // namespace fxrt
