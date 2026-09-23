#include <string>
#include <vector>

#include "ops/ascend/hccl/hccl_all_to_all.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {
bool IsAllToAllV(const ir::TuplePtr& sendNumelList, const ir::TuplePtr& recvNumelList) {
  for (size_t i = 0; i < sendNumelList->Size(); i++) {
    if (sendNumelList->operator[](i)->ToInt() != sendNumelList->operator[](0)->ToInt()) {
      return true;
    }
  }
  for (size_t i = 0; i < recvNumelList->Size(); i++) {
    if (recvNumelList->operator[](i)->ToInt() != recvNumelList->operator[](0)->ToInt()) {
      return true;
    }
  }
  return false;
}

void GetAllToAllVParam(
    const ir::TuplePtr& sendNumelList,
    const ir::TuplePtr& recvNumelList,
    HcclAllToAllVParams* params,
    const uint64_t strideOfDim0) {
  uint64_t offset = 0;
  for (size_t i = 0; i < sendNumelList->Size(); i++) {
    auto count = static_cast<uint64_t>(sendNumelList->operator[](i)->ToInt()) * strideOfDim0;
    params->sendCounts.push_back(count);
    params->sdispls.push_back(offset);
    offset += count;
  }
  offset = 0;
  for (size_t i = 0; i < recvNumelList->Size(); i++) {
    auto count = static_cast<uint64_t>(recvNumelList->operator[](i)->ToInt()) * strideOfDim0;
    params->recvCounts.push_back(count);
    params->rdispls.push_back(offset);
    offset += count;
  }
}

OpsErrorCode HcclAllToAll::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  const string& groupName = input[kIndex3]->ToString();
  auto rankSize = fxrt::collective::CollectiveManager::Instance().GetGroupSize(groupName);
  if (rankSize == 0) {
    RT_GLOG(ERROR) << "HcclAllToAll got zero rank size for group: " << groupName;
    return INVALID_PARAM;
  }
  HcclAdapter::GetInstance().InitHccl();
  auto inputTensor = input[kIndex0]->ToTensor();
  HcomUtil::CheckHcclInputContiguous(inputTensor, "HcclAllToAll");
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(inputTensor);
  hcclKernel_.hcclCount_ = hcclCount / rankSize;
  hcclKernel_.hcclDataType_ = hcclDataType;
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);
  useAllToAllV_ = IsAllToAllV(input[kIndex2]->ToTuple(), input[kIndex1]->ToTuple());
  return SUCCESS;
}

OpsErrorCode HcclAllToAll::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  auto outTensor = output->ToTensor();
  ::HcclResult hcclResult;
  if (useAllToAllV_) {
    uint64_t strideOfDim0 = 1;
    auto& inputShape = input[kIndex0]->ToTensor()->Shape();
    for (size_t i = 1; i < inputShape.size(); ++i) {
      strideOfDim0 *= inputShape[i];
    }
    HcclAllToAllVParams params;
    GetAllToAllVParam(input[kIndex2]->ToTuple(), input[kIndex1]->ToTuple(), &params, strideOfDim0);
    hcclResult = HcclAdapter::GetInstance().HcclAlltoAllV(
        const_cast<void*>(input[kIndex0]->ToTensor()->DataPtr()),
        outTensor->DataPtr(),
        params,
        hcclKernel_.hcclDataType_,
        stream,
        hcclKernel_.comm_);
  } else {
    HcclAllToAllParams params = {hcclKernel_.hcclCount_, hcclKernel_.hcclCount_};
    hcclResult = HcclAdapter::GetInstance().HcclAllToAll(
        const_cast<void*>(input[kIndex0]->ToTensor()->DataPtr()),
        outTensor->DataPtr(),
        params,
        hcclKernel_.hcclDataType_,
        stream,
        hcclKernel_.comm_);
  }

  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclAllToAll failed, hcclResult: " << hcclResult;
    return LAUNCH_OP_FAILED;
  }

  return SUCCESS;
}
FXRT_REG_OP(all_to_all, HcclAllToAll, Ascend);
} // namespace ops
} // namespace fxrt
