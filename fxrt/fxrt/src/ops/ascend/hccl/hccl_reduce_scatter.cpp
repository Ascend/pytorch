#include <string>
#include <vector>
#include "ops/ascend/hccl/hccl_reduce_scatter.h"
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

void HcclReduceScatter::Init(const std::vector<const ir::Value*>& input, const ir::Value* output) {
  hcclOpType_ = HcomUtil::GetHcomReduceOpType(input[kIndex1]->ToString());
}

OpsErrorCode HcclReduceScatter::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  HcclAdapter::GetInstance().InitHccl();
  auto rankSize = input[kIndex2]->ToInt();
  auto inputTensor = input[kIndex0]->ToTensor();
  HcomUtil::CheckHcclInputContiguous(inputTensor, "HcclReduceScatter");
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(inputTensor, rankSize);
  hcclKernel_.hcclCount_ = hcclCount;
  hcclKernel_.hcclDataType_ = hcclDataType;
  const string& groupName = input[kIndex3]->ToString();
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);

  return SUCCESS;
}

OpsErrorCode HcclReduceScatter::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto outTensor = output->ToTensor();

  auto hcclResult = HcclAdapter::GetInstance().HcclReduceScatter(
      const_cast<void*>(input[kIndex0]->ToTensor()->DataPtr()),
      outTensor->DataPtr(),
      hcclKernel_.hcclCount_,
      hcclKernel_.hcclDataType_,
      hcclOpType_,
      stream,
      hcclKernel_.comm_);

  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclReduceScatter failed, hccl_result: " << hcclResult;
  }

  return SUCCESS;
}
FXRT_REG_OP(reduce_scatter, HcclReduceScatter, Ascend);
} // namespace ops
} // namespace fxrt
