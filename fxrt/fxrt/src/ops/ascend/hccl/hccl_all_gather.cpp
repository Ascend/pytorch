#include <vector>
#include <string>

#include "ops/ascend/hccl/hccl_all_gather.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"

#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace fxrt {
namespace ops {
OpsErrorCode HcclAllGather::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  HcclAdapter::GetInstance().InitHccl();
  auto inputTensor = input[kIndex0]->ToTensor();
  HcomUtil::CheckHcclInputContiguous(inputTensor, "HcclAllGather");
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(inputTensor);
  hcclKernel_.hcclCount_ = hcclCount;
  hcclKernel_.hcclDataType_ = hcclDataType;
  const string& groupName = input[kIndex2]->ToString();
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);

  return SUCCESS;
}

OpsErrorCode HcclAllGather::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto hccl_result = HcclAdapter::GetInstance().HcclAllGather(
      const_cast<void*>(input[kIndex0]->ToTensor()->DataPtr()),
      output->ToTensor()->DataPtr(),
      hcclKernel_.hcclCount_,
      hcclKernel_.hcclDataType_,
      stream,
      hcclKernel_.comm_);
  if (hccl_result != ::HcclResult::HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcomAllGather failed, hccl_result: " << hccl_result;
  }

  return SUCCESS;
}
FXRT_REG_OP(all_gather, HcclAllGather, Ascend);
} // namespace ops
} // namespace fxrt
