#include "ops/ascend/hccl/wait_tensor.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
namespace fxrt {
namespace ops {

OpsErrorCode HcclWaitTensor::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  RT_VLOG(VL_OPS) << "WaitTensor InferShape";
  auto& input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto& outputTensor = output->ToTensor();
  auto& outputShape = outputTensor->Shape();
  outputShape = input0Shape;
  outputTensor->Resize();
  return SUCCESS;
}

OpsErrorCode HcclWaitTensor::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  return SUCCESS;
}

OpsErrorCode HcclWaitTensor::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool HcclWaitTensor::NeedLaunch() {
  return false;
}

FXRT_REG_OP(wait_tensor, HcclWaitTensor, Ascend);
} // namespace ops
} // namespace fxrt
