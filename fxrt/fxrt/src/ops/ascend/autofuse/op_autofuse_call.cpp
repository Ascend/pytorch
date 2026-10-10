#include "ops/ascend/autofuse/op_autofuse_call.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/logger.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"

namespace fxrt {
namespace ops {
namespace {

constexpr uint32_t kShimAbiVersion = 1U;

void CheckStringInput(const std::vector<const ir::Value*>& inputs, size_t index, const char* name) {
  if (index >= inputs.size() || inputs[index] == nullptr || !inputs[index]->IsString()) {
    RT_GLOG(EXCEPTION) << "autofuse_call expects input[" << index << "] to be a " << name << " string";
  }
}

} // namespace

OpAutofuseCall::~OpAutofuseCall() {
  CloseWrapper();
}

void OpAutofuseCall::CloseWrapper() {
  if (stubHandle_ == nullptr) {
    return;
  }
  if (finalizeFunc_ != nullptr && context_ != nullptr) {
    const char* key = kernelKey_.empty() ? nullptr : kernelKey_.c_str();
    (void)finalizeFunc_(context_, key);
  }
  context_ = nullptr;
  abiVersionFunc_ = nullptr;
  argNumFunc_ = nullptr;
  initFunc_ = nullptr;
  launchFunc_ = nullptr;
  finalizeFunc_ = nullptr;
  expectedArgNum_ = 0U;
  (void)dlclose(stubHandle_);
  stubHandle_ = nullptr;
}

std::vector<size_t> OpAutofuseCall::ParseMutatedArgIndices(const ir::Value* value) {
  CHECK_IF_NULL(value);
  if (!value->IsTuple()) {
    RT_GLOG(EXCEPTION) << "autofuse_call expects mutated argument positions as a tuple";
  }
  std::vector<size_t> indices;
  const auto& tuple = value->ToTuple();
  indices.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); ++i) {
    const auto* elem = (*tuple)[i].get();
    CHECK_IF_NULL(elem);
    if (!elem->IsInt() || elem->ToInt() < 0) {
      RT_GLOG(EXCEPTION) << "autofuse_call mutated arg index must be a non-negative int, got " << *elem;
    }
    indices.push_back(static_cast<size_t>(elem->ToInt()));
  }
  return indices;
}

std::vector<const ir::Value*> OpAutofuseCall::FlattenOutputs(const ir::Value* output) {
  CHECK_IF_NULL(output);
  if (output->IsTensor()) {
    return {output};
  }
  if (output->IsTuple()) {
    std::vector<const ir::Value*> outputs;
    const auto& tuple = output->ToTuple();
    outputs.reserve(tuple->Size());
    for (size_t i = 0; i < tuple->Size(); ++i) {
      const auto* elem = (*tuple)[i].get();
      CHECK_IF_NULL(elem);
      if (!elem->IsTensor()) {
        RT_GLOG(EXCEPTION) << "autofuse_call output[" << i << "] must be a Tensor";
      }
      outputs.push_back(elem);
    }
    return outputs;
  }
  RT_GLOG(EXCEPTION) << "autofuse_call output must be Tensor or Tuple[Tensor]";
}

uint64_t OpAutofuseCall::EncodeArg(const ir::Value* value, size_t index) {
  CHECK_IF_NULL(value);
  if (value->IsTensor()) {
    return reinterpret_cast<uint64_t>(value->ToTensor()->DataPtr());
  }
  if (value->IsInt() || value->IsSymbol()) {
    return static_cast<uint64_t>(value->ToInt());
  }
  if (value->IsBool()) {
    return value->ToBool() ? 1U : 0U;
  }
  if (value->IsDouble()) {
    uint64_t payload = 0U;
    const double scalar = value->ToDouble();
    static_assert(sizeof(scalar) <= sizeof(payload), "double payload must fit in uint64_t");
    std::memcpy(&payload, &scalar, sizeof(scalar));
    return payload;
  }
  RT_GLOG(EXCEPTION) << "autofuse_call argument[" << index << "] must be Tensor, int, symbol, bool or double, got "
                     << *value;
}

void OpAutofuseCall::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  (void)output;
  if (inputs.size() < kRealInputStartIndex) {
    RT_GLOG(EXCEPTION) << "autofuse_call expects at least " << kRealInputStartIndex << " inputs";
  }
  CheckStringInput(inputs, kStubPathInputIndex, "generated stub path");
  CheckStringInput(inputs, kWrapperPathInputIndex, "wrapper path");
  CheckStringInput(inputs, kKernelPathInputIndex, "kernel path");
  CheckStringInput(inputs, kKernelKeyInputIndex, "kernel key");

  stubPath_ = inputs[kStubPathInputIndex]->ToString();
  wrapperPath_ = inputs[kWrapperPathInputIndex]->ToString();
  kernelPath_ = inputs[kKernelPathInputIndex]->ToString();
  kernelKey_ = inputs[kKernelKeyInputIndex]->ToString();
  realInputs_.assign(inputs.begin() + kRealInputStartIndex, inputs.end());

  const auto mutatedArgIndices = ParseMutatedArgIndices(inputs[kMutatedArgIndicesInputIndex]);
  const size_t argNum = realInputs_.size();
  refPairs_.clear();
  std::vector<bool> seen(argNum, false);
  for (size_t outputIndex = 0; outputIndex < mutatedArgIndices.size(); ++outputIndex) {
    const size_t argIndex = mutatedArgIndices[outputIndex];
    if (argIndex >= argNum || seen[argIndex]) {
      RT_GLOG(EXCEPTION) << "autofuse_call mutated arg index is invalid or duplicated: " << argIndex;
    }
    seen[argIndex] = true;
    if (!realInputs_[argIndex]->IsTensor()) {
      RT_GLOG(EXCEPTION) << "autofuse_call mutated arg[" << argIndex << "] must be a Tensor";
    }
    refPairs_.emplace_back(static_cast<uint32_t>(outputIndex), static_cast<uint32_t>(kRealInputStartIndex + argIndex));
  }
  if (refPairs_.empty()) {
    RT_GLOG(EXCEPTION) << "autofuse_call requires at least one mutated output";
  }
  if (FlattenOutputs(output).size() != refPairs_.size()) {
    RT_GLOG(EXCEPTION) << "autofuse_call output count does not match mutated argument count";
  }

  stubHandle_ = dlopen(stubPath_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (stubHandle_ == nullptr) {
    RT_GLOG(EXCEPTION) << "Failed to load generated AutoFuse wrapper shim " << stubPath_ << ": " << dlerror();
  }
  abiVersionFunc_ = reinterpret_cast<AbiVersionFunc>(dlsym(stubHandle_, "fxrt_autofuse_abi_version"));
  argNumFunc_ = reinterpret_cast<ArgNumFunc>(dlsym(stubHandle_, "fxrt_autofuse_arg_num"));
  initFunc_ = reinterpret_cast<InitFunc>(dlsym(stubHandle_, "fxrt_autofuse_init"));
  launchFunc_ = reinterpret_cast<LaunchFunc>(dlsym(stubHandle_, "fxrt_autofuse_launch"));
  finalizeFunc_ = reinterpret_cast<FinalizeFunc>(dlsym(stubHandle_, "fxrt_autofuse_finalize"));
  if (abiVersionFunc_ == nullptr || argNumFunc_ == nullptr || initFunc_ == nullptr || launchFunc_ == nullptr ||
      finalizeFunc_ == nullptr) {
    CloseWrapper();
    RT_GLOG(EXCEPTION) << "Generated AutoFuse wrapper shim has an incomplete fixed ABI: " << stubPath_;
  }
  if (abiVersionFunc_() != kShimAbiVersion) {
    CloseWrapper();
    RT_GLOG(EXCEPTION) << "Unsupported generated AutoFuse wrapper shim ABI version: " << stubPath_;
  }
  expectedArgNum_ = argNumFunc_();
  if (expectedArgNum_ != argNum) {
    CloseWrapper();
    RT_GLOG(EXCEPTION) << "Generated AutoFuse wrapper shim expects " << expectedArgNum_ << " arguments, graph supplied "
                       << argNum;
  }

  const char* key = kernelKey_.empty() ? nullptr : kernelKey_.c_str();
  const int64_t ret = initFunc_(&context_, wrapperPath_.c_str(), kernelPath_.c_str(), key);
  if (ret != 0 || context_ == nullptr) {
    CloseWrapper();
    RT_GLOG(EXCEPTION) << "Generated AutoFuse wrapper shim init failed, return code=" << ret
                       << ", wrapper=" << wrapperPath_ << ", kernel=" << kernelPath_;
  }
  RT_GLOG(INFO) << "OpAutofuseCall initialized, stub=" << stubPath_ << ", wrapper=" << wrapperPath_
                << ", kernel=" << kernelPath_ << ", args=" << argNum << ", mutated=" << refPairs_.size();
}

OpsErrorCode OpAutofuseCall::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  (void)input;
  return Operator::InferShape(realInputs_, output);
}

OpsErrorCode OpAutofuseCall::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  (void)input;
  CHECK_IF_NULL(workspaceSize);
  // A mutation output aliases the preallocated buffer passed at its mutated
  // argument position. Generic ref handling only propagates strides and
  // storage offset, so copy the remaining runtime layout metadata as well.
  SyncOutputMetadata(output);

  // wrapper.so owns tiling and workspace allocation. Invoke it inline before
  // the executor queues later ops. In pipeline mode an ordinary Launch is
  // submitted through torch-npu's task queue, while wrapper() submits the real
  // AutoFuse launch through that same queue. That nested submission would put
  // the kernel behind consumers such as Gather. A null stream matches the
  // Python launcher; wrapper.so resolves torch-npu's current stream itself.
  *workspaceSize = 0;
  return LaunchWrapper(nullptr);
}

void OpAutofuseCall::SyncOutputMetadata(const ir::Value* output) const {
  const auto outputs = FlattenOutputs(output);
  for (const auto& refPair : refPairs_) {
    const auto& outputTensor = outputs[refPair.first]->ToTensor();
    const auto& argTensor = realInputs_[refPair.second - kRealInputStartIndex]->ToTensor();
    CHECK_IF_NULL(outputTensor);
    CHECK_IF_NULL(argTensor);
    outputTensor->SetStorageShape(argTensor->StorageShape());
    outputTensor->SetFormat(argTensor->Format());
  }
}

OpsErrorCode OpAutofuseCall::LaunchWrapper(void* stream) {
  if (launchFunc_ == nullptr || context_ == nullptr) {
    RT_GLOG(ERROR) << "Generated AutoFuse wrapper shim is not ready for launch";
    return LAUNCH_OP_FAILED;
  }
  std::vector<uint64_t> args;
  args.reserve(realInputs_.size());
  for (size_t i = 0; i < realInputs_.size(); ++i) {
    args.push_back(EncodeArg(realInputs_[i], i));
  }
  const char* key = kernelKey_.empty() ? nullptr : kernelKey_.c_str();
  const int64_t ret = launchFunc_(context_, args.data(), static_cast<uint32_t>(args.size()), stream, key);
  if (ret != 0) {
    RT_GLOG(ERROR) << "Generated AutoFuse wrapper shim launch failed, return code=" << ret;
    return LAUNCH_OP_FAILED;
  }
  return SUCCESS;
}

OpsErrorCode OpAutofuseCall::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  (void)input;
  (void)workspace;
  (void)workspaceSize;
  (void)output;
  // NeedLaunch() is false because wrapper() is submitted from CalcWorkspace.
  // Retain Launch for direct/manual operator invocations.
  return LaunchWrapper(stream);
}

} // namespace ops
} // namespace fxrt
