#include "ops/utils/op_support.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>

#include "common/logger.h"
#include "hardware/device.h"
#include "ops/op_register.h"

namespace fxrt {
namespace runtime {

namespace {

bool IsOpSupportWhitelisted(const std::string& opName) {
  // Ops listed in ops.list for FxConverter Path C / identity bypass but without Ascend
  // Aclnn FXRT_REG_OP. Whitelist so FX→IR keeps the logical Op for converter; runtime
  // Launch goes through Path C / IdentityAlias, not an Aclnn stub.
  static const std::unordered_set<std::string> whitelist = {
      "make_tuple",
      "tuple_getitem",
      "depend",
      "return",
      "update_state",
      "gather",
      "constant_pad_nd",
      "full_like",
      "clone",
      "contiguous",
      "embedding",
      // ViewAlias (GE Reshape metadata): same device addr, new StorageShape.
      "view",
      "reshape",
      "unsqueeze_view",
      "unsqueeze",
      "squeeze_view",
      "squeeze",
      "squeeze_dim",
      "flatten_view",
      "alias",
      // Path C Transpose materialization (not ViewAlias).
      "permute_view",
      "permute",
      // Path C SliceV2 materialization (contiguous step==1; not ViewAlias).
      "slice_view",
      "slice",
      "narrow_view",
      "narrow",
      // getitem / as_strided contiguous degrade / empty contiguous alloc (converter Path C / ViewAlias).
      "getitem_slice",
      "gather_v2",
      "index_tensor",
      "as_strided_view",
      "empty_strided",
  };
  return whitelist.find(opName) != whitelist.end();
}

// For all-tensor ops: verify all actual inputs are tensors and count matches prototype.
struct DialectOpInfo {
  size_t tensorInputCount{0};
};

bool GetDialectOpInfo(const std::string& opName, DialectOpInfo* info) {
  const size_t* tensorInputCount = fxrt::ops::OpPrototypeRegistry::GetTensorInputCount(opName);
  if (tensorInputCount == nullptr) {
    return false;
  }
  info->tensorInputCount = *tensorInputCount;
  return true;
}

bool IsOpRegisteredOnDevice(const std::string& opName, const hardware::DeviceType deviceType) {
  if (deviceType == hardware::DeviceType::NPU) {
    return fxrt::ops::OpFactory<fxrt::ops::Operator>::GetInstance().IsRegistered(opName);
  }
  if (deviceType == hardware::DeviceType::CPU) {
    return fxrt::ops::OpFactory<fxrt::ops::Operator, fxrt::ops::CPUOpFactory>::GetInstance().IsRegistered(opName);
  }
  return false;
}

// For all-tensor ops: verify all actual inputs are tensors and count matches prototype.
OpSupportResult CheckInputTypesSupported(const std::string& opName, const std::vector<ir::ValuePtr>& inputValues) {
  DialectOpInfo info;
  if (!GetDialectOpInfo(opName, &info)) {
    OpSupportResult r;
    r.status = OpSupportStatus::kOk;
    r.message.clear();
    return r;
  }

  // Some graphs carry control-edge dependencies as `None` values (e.g. output of `End`).
  // For all-fixed-tensor ops, these `None` inputs are not real data inputs and should be ignored.
  std::vector<ir::ValuePtr> filteredInputs;
  filteredInputs.reserve(inputValues.size());
  for (const auto& v : inputValues) {
    if (v != nullptr && v->IsNone()) {
      continue;
    }
    filteredInputs.push_back(v);
  }

  // 1. Verify all actual inputs are tensors (no scalar, tuple, etc.)
  for (size_t i = 0; i < filteredInputs.size(); ++i) {
    const auto& v = filteredInputs[i];
    if (v == nullptr || !v->IsTensor()) {
      OpSupportResult r;
      r.status = OpSupportStatus::kUnsupportedInputType;
      r.message =
          "operator '" + opName + "' expects all tensor inputs, but input[" + std::to_string(i) + "] is not a tensor";
      return r;
    }
  }

  // 2. Verify input count matches prototype (exact match for all-fixed-tensor ops)
  size_t actualCount = filteredInputs.size();
  if (actualCount != info.tensorInputCount) {
    OpSupportResult r;
    r.status = OpSupportStatus::kUnsupportedInputType;
    r.message = "operator '" + opName + "' expects " + std::to_string(info.tensorInputCount) +
        " tensor input(s), got " + std::to_string(actualCount);
    return r;
  }

  OpSupportResult r;
  r.status = OpSupportStatus::kOk;
  r.message.clear();
  return r;
}

} // namespace

hardware::Device GetDeviceFromOutputAndInputs(const ir::ValuePtr& output, const std::vector<ir::ValuePtr>& inputs) {
  CHECK_IF_NULL(output);

  if (output->IsTensor()) {
    auto& tensor = output->ToTensor();
    CHECK_IF_NULL(tensor);
    return tensor->GetDevice();
  }

  if (output->IsNone()) {
    auto it =
        std::find_if(inputs.begin(), inputs.end(), [](const ir::ValuePtr& v) { return v != nullptr && v->IsTensor(); });
    if (it != inputs.end()) {
      return (*it)->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  if (output->IsTuple()) {
    auto& tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);

    if (tuple->Size() == 0) {
      return {hardware::DeviceType::CPU, 0};
    }

    bool allTensor = std::all_of(
        tuple->begin(), tuple->end(), [](const ir::ValuePtr& elem) { return elem != nullptr && elem->IsTensor(); });

    if (allTensor) {
      return (*tuple->begin())->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  return {hardware::DeviceType::CPU, 0};
}

OpSupportResult CheckOpSupport(
    const std::string& opName,
    const ir::ValuePtr& outputValue,
    const std::vector<ir::ValuePtr>& inputValues) {
  if (IsOpSupportWhitelisted(opName)) {
    OpSupportResult r;
    r.status = OpSupportStatus::kOk;
    r.message.clear();
    return r;
  }

  const hardware::Device device = GetDeviceFromOutputAndInputs(outputValue, inputValues);
  if (!IsOpRegisteredOnDevice(opName, device.type)) {
    OpSupportResult r;
    r.status = OpSupportStatus::kUnsupportedDevice;
    r.message = "operator '" + opName + "' not registered on target device '" +
        hardware::GetDeviceNameByType(device.type) + "'";
    return r;
  }

  return CheckInputTypesSupported(opName, inputValues);
}

} // namespace runtime
} // namespace fxrt
