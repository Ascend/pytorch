#include "ops/ascend/hccl/tensor_copy.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "hardware/ascend/res_manager/ascend_res_manager.h"

#include "common/logger.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode HcclTensorCopy::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  RT_VLOG(VL_OPS) << "TensorCopy InferShape";
  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto& outputTensor = output->ToTensor();
  auto& outputShape = outputTensor->Shape();
  outputShape = input0Shape;
  outputTensor->Resize();
  return SUCCESS;
}

OpsErrorCode HcclTensorCopy::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  return SUCCESS;
}

OpsErrorCode HcclTensorCopy::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto srcTensor = input[kIndex1]->ToTensor();
  auto outTensor = input[kIndex0]->ToTensor();
  auto dstSize = outTensor->Numel() * outTensor->Dtype().GetSize();

  // host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr
  auto ret = fxrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(
      outTensor->DataPtr(), dstSize, srcTensor->DataPtr(), dstSize, stream);
  if (ret == false) {
    RT_GLOG(ERROR) << " call aclrtMemcpyAsync in Op TensorCopy failed";
  }

  return SUCCESS;
}
FXRT_REG_OP(copy, HcclTensorCopy, Ascend);
} // namespace ops
} // namespace fxrt
