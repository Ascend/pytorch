#include <string>
#include <vector>

#include "ops/ascend/hccl/hccl_all_reduce.h"
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

void HcclAllReduce::Init(const std::vector<const ir::Value*>& input, const ir::Value* output) {
  hcclOpType_ = HcomUtil::GetHcomReduceOpType(input[kIndex1]->ToString());
}

OpsErrorCode HcclAllReduce::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  HcclAdapter::GetInstance().InitHccl();
  auto inputTensor = input[kIndex0]->ToTensor();
  HcomUtil::CheckHcclInputContiguous(inputTensor, "HcclAllReduce");
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(inputTensor);
  hcclKernel_.hcclCount_ = hcclCount;
  hcclKernel_.hcclDataType_ = hcclDataType;
  const string& groupName = input[kIndex2]->ToString();
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);

  return SUCCESS;
}

OpsErrorCode HcclAllReduce::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto outTensor = output->ToTensor();

  auto hcclResult = HcclAdapter::GetInstance().HcclAllReduce(
      const_cast<void*>(input[kIndex0]->ToTensor()->DataPtr()),
      outTensor->DataPtr(),
      hcclKernel_.hcclCount_,
      hcclKernel_.hcclDataType_,
      hcclOpType_,
      stream,
      hcclKernel_.comm_);

  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclAllReduce failed, hcclResult: " << hcclResult;
  }

  return SUCCESS;
}
FXRT_REG_OP(all_reduce, HcclAllReduce, Ascend);
} // namespace ops
} // namespace fxrt
