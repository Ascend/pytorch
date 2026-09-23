#include "ops/ascend/dvm/op_dvm_call_v2.h"

#include <ATen/record_function.h>
#include <pybind11/detail/common.h>
#include <pybind11/detail/type_caster_base.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "common/logger.h"
#include "hardware/hardware_abstract/memory_manager.h"
#include "hardware/device.h"
#include "ir/tensor/storage.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "ir/tensor/format.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "torch_npu/csrc/inductor/dvm/pybind_api.h"

namespace py = pybind11;

namespace fxrt {
namespace ops {
namespace dvm_v2 {
namespace {

int64_t ScalarInputValue(const ir::Value* value, size_t index) {
  CHECK_IF_NULL(value);
  if (value->IsInt() || value->IsSymbol()) {
    return value->ToInt();
  }
  if (value->IsBool()) {
    return value->ToBool() ? 1 : 0;
  }
  RT_GLOG(EXCEPTION) << "DVM V2 dynamic scalar input[" << index << "] must be int/bool/symbol, got: " << *value;
}

dvm::DynKernelPy* AsDynKernel(py::handle obj) {
  if (obj.is_none() || obj.ptr() == nullptr) {
    RT_GLOG(EXCEPTION) << "DVM V2 dynamic kernel object is null.";
  }
  if (!py::detail::is_holder_constructed(obj.ptr())) {
    RT_GLOG(EXCEPTION) << "DVM V2 dynamic kernel object is not a pybind11 holder instance.";
  }
  auto* inst = reinterpret_cast<py::detail::instance*>(obj.ptr());
  auto vh = inst->get_value_and_holder();
  auto* value = vh.value_ptr();
  CHECK_IF_NULL(value);
  return static_cast<dvm::DynKernelPy*>(value);
}

void UpdateSymShapeData(dvm::DynKernelPy* kernel) {
  CHECK_IF_NULL(kernel);
  const auto& shapeRefs = kernel->ShapeRefs();
  const auto& symShapeRefs = kernel->SymShapeRefs();
  if (shapeRefs.size() != symShapeRefs.size()) {
    RT_GLOG(EXCEPTION) << "DVM V2 dynamic symbolic shape ref count mismatch, shape refs " << shapeRefs.size()
                       << ", symbolic shape refs " << symShapeRefs.size();
  }
  for (size_t i = 0; i < symShapeRefs.size(); ++i) {
    CHECK_IF_NULL(shapeRefs[i]);
    if (shapeRefs[i]->size != symShapeRefs[i].size()) {
      RT_GLOG(EXCEPTION) << "DVM V2 dynamic symbolic shape rank mismatch at ref " << i << ", shape rank "
                         << shapeRefs[i]->size << ", symbolic rank " << symShapeRefs[i].size();
    }
    for (size_t j = 0; j < symShapeRefs[i].size(); ++j) {
      shapeRefs[i]->shape_data[j] = symShapeRefs[i][j]->data_.i64;
    }
  }
}

} // namespace

void UpdateDynamicKernelRefs(
    const py::object& kernelObj,
    const std::vector<const ir::Value*>& inputs,
    std::vector<std::vector<int64_t>>* inputShapes,
    std::vector<std::vector<int64_t>>* inputStrides) {
  CHECK_IF_NULL(inputShapes);
  CHECK_IF_NULL(inputStrides);
  auto* kernel = AsDynKernel(kernelObj);
  CHECK_IF_NULL(kernel);

  const auto& loadShapeRefs = kernel->DynLoadShapeRefs();
  auto& symScalarInputs = kernel->SymScalarInputs();
  const size_t numTensorInputs = loadShapeRefs.size();
  const size_t numScalarInputs = symScalarInputs.size();
  if (inputs.size() != numTensorInputs + numScalarInputs) {
    RT_GLOG(EXCEPTION) << "DVM V2 dynamic input count mismatch, expects " << numTensorInputs << " tensor inputs and "
                       << numScalarInputs << " scalar inputs, got " << inputs.size();
  }

  inputShapes->clear();
  inputStrides->clear();
  inputShapes->reserve(numTensorInputs);
  inputStrides->reserve(numTensorInputs);

  size_t tensorIndex = 0;
  size_t scalarIndex = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto* input = inputs[i];
    CHECK_IF_NULL(input);
    if (input->IsTensor()) {
      if (tensorIndex >= numTensorInputs) {
        RT_GLOG(EXCEPTION) << "DVM V2 dynamic got more tensor inputs than kernel loads, input index: " << i;
      }
      const auto& tensor = input->ToTensor();
      inputShapes->push_back(tensor->Shape());
      inputStrides->push_back(tensor->Strides());
      auto* shapeRef = loadShapeRefs[tensorIndex];
      CHECK_IF_NULL(shapeRef);
      shapeRef->shape.data = (*inputShapes)[tensorIndex].data();
      shapeRef->shape.size = (*inputShapes)[tensorIndex].size();
      if (shapeRef->stride.data != nullptr) {
        shapeRef->stride.data = (*inputStrides)[tensorIndex].data();
        shapeRef->stride.size = (*inputStrides)[tensorIndex].size();
      }
      ++tensorIndex;
      continue;
    }

    if (scalarIndex >= numScalarInputs) {
      RT_GLOG(EXCEPTION) << "DVM V2 dynamic got more scalar inputs than kernel scalar refs, input index: " << i;
    }
    symScalarInputs[scalarIndex]->data_ = ScalarInputValue(input, i);
    ++scalarIndex;
  }

  if (tensorIndex != numTensorInputs || scalarIndex != numScalarInputs) {
    RT_GLOG(EXCEPTION) << "DVM V2 dynamic input mismatch after parsing, tensors " << tensorIndex << "/"
                       << numTensorInputs << ", scalars " << scalarIndex << "/" << numScalarInputs;
  }

  UpdateSymShapeData(kernel);
}

} // namespace dvm_v2

namespace {

constexpr size_t kHandleInputIndex = 0;
constexpr size_t kRealInputStartIndex = 1;

struct WorkspaceSizeRecorder : public dvm::WsAllocator {
  explicit WorkspaceSizeRecorder(size_t* size) : size_(size) {}
  void* Alloc(size_t size) override {
    CHECK_IF_NULL(size_);
    *size_ = size;
    return nullptr;
  }
  size_t* size_ = nullptr;
};

struct ExternalWorkspaceAllocator : public dvm::WsAllocator {
  explicit ExternalWorkspaceAllocator(void* workspace) : workspace_(workspace) {}
  void* Alloc(size_t) override {
    return workspace_;
  }
  void* workspace_ = nullptr;
};

bool IsGroupLaunchEnabled() {
  const char* value = std::getenv("FXRT_KERNEL_LAUNCH_GROUP_NUM");
  return value != nullptr && std::string_view(value).empty() == false;
}

ir::DataType DvmDTypeToFxrtDType(dvm::DataType dtype) {
  switch (dtype) {
    case dvm::kBool:
      return ir::DataType::Bool;
    case dvm::kFloat16:
      return ir::DataType::Float16;
    case dvm::kBFloat16:
      return ir::DataType::BFloat16;
    case dvm::kFloat32:
      return ir::DataType::Float32;
    case dvm::kInt32:
      return ir::DataType::Int32;
    case dvm::kInt64:
      return ir::DataType::Int64;
    default:
      RT_GLOG(EXCEPTION) << "Unsupported DVM output dtype: " << static_cast<int>(dtype);
  }
}

std::vector<ir::TensorPtr> FlattenTensors(const ir::Value* value) {
  CHECK_IF_NULL(value);
  if (value->IsTensor()) {
    return {value->ToTensor()};
  }
  if (value->IsTuple()) {
    return value->ToTuple()->ToTensorList();
  }
  RT_GLOG(EXCEPTION) << "DVM V2 output must be Tensor or Tuple[Tensor], got: " << *value;
}

bool IsDvmSupportedDenseFormat(ir::MemoryFormat format) {
  return format == ir::FORMAT_ND || format == ir::FORMAT_NCHW;
}

bool IsContiguousTensor(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  const auto& shape = tensor->Shape();
  const auto& strides = tensor->Strides();
  if (strides.empty()) {
    return true;
  }
  if (shape.size() != strides.size()) {
    return false;
  }
  int64_t cumulatedStride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 0) {
      return true;
    }
    if (shape[i] != 1 && strides[i] != cumulatedStride) {
      return false;
    }
    cumulatedStride *= shape[i];
  }
  return true;
}

bool IsProvenDenseContiguousTensor(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  if (!IsDvmSupportedDenseFormat(tensor->Format())) {
    return false;
  }
  if (tensor->StorageOffset() < 0) {
    return false;
  }

  const auto& shape = tensor->Shape();
  const auto& strides = tensor->Strides();
  if (strides.empty()) {
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
  }
  if (shape.size() != strides.size()) {
    return false;
  }

  int64_t expectedStride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] < 0 || strides[i] < 0) {
      return false;
    }
    if (shape[i] == 0) {
      return true;
    }
    if (shape[i] != 1 && strides[i] != expectedStride) {
      return false;
    }
    expectedStride *= shape[i];
  }
  return true;
}

size_t AlignWorkspaceSize(size_t size) {
  if (size == 0) {
    return 0;
  }
  return ((size + device::kMemAlignSize - 1) / device::kMemAlignSize) * device::kMemAlignSize;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] < 0) {
      RT_GLOG(EXCEPTION) << "DVM V2 cannot build a contiguous temporary tensor for dynamic dimension: " << shape;
    }
    stride *= shape[i];
  }
  return strides;
}

size_t TensorDataSizeBytes(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  if (tensor->Numel() < 0) {
    RT_GLOG(EXCEPTION) << "DVM V2 runtime contiguous copy requires concrete input shape, got shape=" << tensor->Shape();
  }
  return static_cast<size_t>(tensor->Numel()) * tensor->Dtype().GetSize();
}

ir::TensorPtr MakeWorkspaceTensorLike(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  auto storage = ir::MakeIntrusive<ir::Storage>(nullptr, TensorDataSizeBytes(tensor), tensor->GetDevice());
  auto temp = ir::MakeIntrusive<ir::Tensor>(storage, tensor->Shape(), tensor->Dtype());
  temp->SetStrides(MakeContiguousStrides(tensor->Shape()));
  temp->SetStorageOffset(0);
  temp->SetStorageShape(tensor->Shape());
  temp->SetFormat(ir::FORMAT_ND);
  temp->SetOwnsStorage(false);
  return temp;
}

void CheckTensorSupported(
    const ir::TensorPtr& tensor,
    const char* role,
    size_t index,
    bool requireContiguous,
    bool requireZeroStorageOffset) {
  CHECK_IF_NULL(tensor);
  if (tensor->GetDevice().type != hardware::DeviceType::NPU) {
    RT_GLOG(EXCEPTION) << "DVM V2 only supports NPU tensors, but " << role << "[" << index << "] is on "
                       << hardware::GetDeviceNameByType(tensor->GetDevice().type);
  }
  if (requireContiguous && !IsContiguousTensor(tensor)) {
    RT_GLOG(EXCEPTION) << "DVM V2 only supports contiguous tensors, but " << role << "[" << index
                       << "] has shape=" << tensor->Shape() << " and strides=" << tensor->Strides();
  }
  if (!IsDvmSupportedDenseFormat(tensor->Format())) {
    RT_GLOG(EXCEPTION) << "DVM V2 only supports dense FORMAT_ND or FORMAT_NCHW tensors, but " << role << "[" << index
                       << "] format=" << ir::FormatEnumToStr(tensor->Format());
  }
  if (requireZeroStorageOffset && tensor->StorageOffset() != 0) {
    RT_GLOG(EXCEPTION) << "DVM V2 does not support non-zero storage offset for " << role << "[" << index
                       << "], storageOffset=" << tensor->StorageOffset();
  }
}

bool NeedRuntimeContiguousCopy(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  return !IsProvenDenseContiguousTensor(tensor);
}

void* OffsetWorkspace(void* workspace, size_t offset) {
  if (workspace == nullptr) {
    return nullptr;
  }
  return static_cast<void*>(static_cast<uint8_t*>(workspace) + offset);
}

} // namespace

OpDvmCallV2::ContiguousCopyPlan::ContiguousCopyPlan() = default;
OpDvmCallV2::ContiguousCopyPlan::~ContiguousCopyPlan() = default;
OpDvmCallV2::ContiguousCopyPlan::ContiguousCopyPlan(ContiguousCopyPlan&&) noexcept = default;
OpDvmCallV2::ContiguousCopyPlan& OpDvmCallV2::ContiguousCopyPlan::operator=(ContiguousCopyPlan&&) noexcept = default;

OpDvmCallV2::~OpDvmCallV2() {
  rawKernel_ = nullptr;
  relocs_ = nullptr;
  loads_ = nullptr;
  stores_ = nullptr;
  contiguousCopyPlans_.clear();
  if (kernelObj_ != nullptr) {
    py::gil_scoped_acquire gil;
    kernelObj_.reset();
  }
}

void OpDvmCallV2::RefreshKernelState(const py::object& kernelObj) {
  rawKernel_ = reinterpret_cast<dvm::Kernel*>(kernelObj.attr("kernel")().cast<uintptr_t>());
  relocs_ = reinterpret_cast<std::vector<dvm::RelocEntry>*>(kernelObj.attr("relocs")().cast<uintptr_t>());
  loads_ = reinterpret_cast<std::vector<dvm::NDObject*>*>(kernelObj.attr("loads")().cast<uintptr_t>());
  stores_ = reinterpret_cast<std::vector<dvm::NDObject*>*>(kernelObj.attr("stores")().cast<uintptr_t>());
  numTensorInputs_ = kernelObj.attr("num_tensor_inputs")().cast<size_t>();
  numOutputs_ = kernelObj.attr("num_outputs")().cast<size_t>();
  workspaceSize_ = kernelObj.attr("workspace_size")().cast<size_t>();
  isDynamic_ = kernelObj.attr("is_dynamic")().cast<bool>();
  isSplit_ = kernelObj.attr("is_split")().cast<bool>();
  CHECK_IF_NULL(rawKernel_);
  CHECK_IF_NULL(relocs_);
  CHECK_IF_NULL(loads_);
  CHECK_IF_NULL(stores_);
}

void OpDvmCallV2::ResolveKernelObject(const std::string& handle) {
  py::gil_scoped_acquire gil;
  auto fxBackend = py::module::import("fxrt.dvm_adapter");
  py::object kernelObj = fxBackend.attr("get_dvm_kernel_obj")(handle);
  if (kernelObj.is_none()) {
    RT_GLOG(EXCEPTION) << "DVM V2 function did not produce a kernel object for handle: " << handle;
  }
  kernelObj_ = std::make_unique<py::object>(std::move(kernelObj));
  RefreshKernelState(*kernelObj_);
}

void OpDvmCallV2::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  (void)output;
  if (IsGroupLaunchEnabled()) {
    RT_GLOG(EXCEPTION) << "DVM V2 does not support FXRT group-launch execution mode.";
  }
  if (inputs.empty() || inputs[kHandleInputIndex] == nullptr || !inputs[kHandleInputIndex]->IsString()) {
    RT_GLOG(EXCEPTION) << "DVM V2 expects input[0] to be a registered dvm_func handle string.";
  }
  handle_ = inputs[kHandleInputIndex]->ToString();
  ResolveKernelObject(handle_);

  realInputs_.assign(inputs.begin() + kRealInputStartIndex, inputs.end());
  RT_GLOG(INFO) << "OpDvmCallV2 initialized, handle=" << handle_ << ", tensor_inputs=" << numTensorInputs_
                << ", outputs=" << numOutputs_ << ", dynamic=" << isDynamic_ << ", split=" << isSplit_;
}

void UpdateRelocAddrs(std::vector<dvm::RelocEntry>& relocs, const std::vector<void*>& addrs) {
  CHECK_IF_FAIL(relocs.size() == addrs.size());
  for (size_t i = 0; i < relocs.size(); ++i) {
    relocs[i].addr = addrs[i];
  }
}

void OpDvmCallV2::UpdateOutputMetadata(ir::Value* output) const {
  auto outputTensors = FlattenTensors(output);
  if (outputTensors.size() != numOutputs_) {
    RT_GLOG(EXCEPTION) << "DVM V2 output count mismatch, kernel expects " << numOutputs_ << ", got "
                       << outputTensors.size();
  }
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    auto& tensor = outputTensors[i];
    CheckTensorSupported(tensor, "output", i, true, true);
    auto* shape_ref = rawKernel_->GetShape((*stores_)[i]);
    CHECK_IF_NULL(shape_ref);
    std::vector<int64_t> dvmShape(shape_ref->data, shape_ref->data + shape_ref->size);
    if (isDynamic_) {
      tensor->SetShape(dvmShape);
      tensor->SetDtype(DvmDTypeToFxrtDType(rawKernel_->GetDType((*stores_)[i])));
      tensor->Resize();
      continue;
    }
    if (tensor->Shape() != dvmShape) {
      RT_GLOG(EXCEPTION) << "DVM V2 output shape mismatch for output[" << i << "], FXRT shape=" << tensor->Shape()
                         << ", DVM shape=" << dvmShape << ". Dynamic output resizing is not enabled in this path.";
    }
    tensor->SetDtype(DvmDTypeToFxrtDType(rawKernel_->GetDType((*stores_)[i])));
  }
}

void OpDvmCallV2::UpdateDynamicShapeRefs() {
  py::gil_scoped_acquire gil;
  CHECK_IF_NULL(kernelObj_);
  dvm_v2::UpdateDynamicKernelRefs(*kernelObj_, realInputs_, &dynamicInputShapes_, &dynamicInputStrides_);
}

OpsErrorCode OpDvmCallV2::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  (void)input;
  if (isDynamic_) {
    UpdateDynamicShapeRefs();
    rawKernel_->Normalize();
  }
  UpdateOutputMetadata(output);
  return Operator::InferShape(realInputs_, output);
}

std::vector<void*> OpDvmCallV2::BuildAddressVector(
    const std::vector<const ir::Value*>& inputs,
    ir::Value* output,
    const std::vector<void*>* inputAddrOverrides,
    bool allowNonContiguousInputs) const {
  CHECK_IF_NULL(rawKernel_);
  std::vector<void*> addrs;
  addrs.reserve(numTensorInputs_ + numOutputs_);

  size_t tensorInputIndex = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto* input = inputs[i];
    CHECK_IF_NULL(input);
    if (!input->IsTensor()) {
      continue;
    }
    auto tensor = input->ToTensor();
    CheckTensorSupported(tensor, "input", i, !allowNonContiguousInputs, !allowNonContiguousInputs);
    if (inputAddrOverrides != nullptr && tensorInputIndex < inputAddrOverrides->size() &&
        (*inputAddrOverrides)[tensorInputIndex] != nullptr) {
      addrs.push_back((*inputAddrOverrides)[tensorInputIndex]);
    } else {
      addrs.push_back(tensor->DataPtr());
    }
    ++tensorInputIndex;
  }

  auto outputTensors = FlattenTensors(output);
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    auto& tensor = outputTensors[i];
    CheckTensorSupported(tensor, "output", i, true, true);
    addrs.push_back(tensor->DataPtr());
  }

  const size_t expected = numTensorInputs_ + numOutputs_;
  if (addrs.size() != expected) {
    RT_GLOG(EXCEPTION) << "DVM V2 relocation address count mismatch, expected " << expected << ", got " << addrs.size();
  }
  return addrs;
}

size_t OpDvmCallV2::PlanContiguousInputs(size_t offset) {
  contiguousCopyPlans_.clear();
  size_t tensorInputIndex = 0;
  for (size_t realInputIndex = 0; realInputIndex < realInputs_.size(); ++realInputIndex) {
    const auto* input = realInputs_[realInputIndex];
    CHECK_IF_NULL(input);
    if (!input->IsTensor()) {
      continue;
    }
    auto tensor = input->ToTensor();
    CheckTensorSupported(tensor, "input", tensorInputIndex, false, false);
    if (!NeedRuntimeContiguousCopy(tensor)) {
      ++tensorInputIndex;
      continue;
    }
    RT_GLOG(INFO) << " tensor format:" << tensor->Format() << " shape:" << tensor->Shape()
                  << " stride:" << tensor->Strides() << " offset:" << tensor->StorageOffset();
    ContiguousCopyPlan plan;
    plan.realInputIndex = realInputIndex;
    plan.inputIndex = tensorInputIndex;
    plan.tempTensor = MakeWorkspaceTensorLike(tensor);

    uint64_t copyWorkspaceSize = 0;
    AclnnExecutor executor("aclnnInplaceCopy");
    {
      RECORD_FUNCTION("DVM::GetWorkspaceSize", std::vector<c10::IValue>({}));
      executor.GetWorkspaceSize(&copyWorkspaceSize, plan.tempTensor, tensor);
    }
    plan.copyWorkspaceOffset = offset;
    plan.copyWorkspaceSize = static_cast<size_t>(copyWorkspaceSize);
    offset += AlignWorkspaceSize(plan.copyWorkspaceSize);

    plan.tempBufferOffset = offset;
    plan.tempBufferSize = TensorDataSizeBytes(tensor);
    offset += AlignWorkspaceSize(plan.tempBufferSize);

    contiguousCopyPlans_.emplace_back(std::move(plan));
    ++tensorInputIndex;
  }
  return offset;
}

std::vector<void*> OpDvmCallV2::PrepareContiguousInputs(void* workspace, size_t workspaceSize, void* stream) {
  std::vector<void*> inputAddrOverrides(numTensorInputs_, nullptr);
  for (auto& plan : contiguousCopyPlans_) {
    CHECK_IF_FAIL(plan.inputIndex < numTensorInputs_);
    CHECK_IF_FAIL(plan.realInputIndex < realInputs_.size());
    auto tensor = realInputs_[plan.realInputIndex]->ToTensor();
    CheckTensorSupported(tensor, "input", plan.inputIndex, false, false);

    if (!NeedRuntimeContiguousCopy(tensor)) {
      continue;
    }

    const size_t tempBufferEnd = plan.tempBufferOffset + plan.tempBufferSize;
    if (tempBufferEnd > workspaceSize) {
      RT_GLOG(EXCEPTION) << "DVM V2 contiguous input temporary buffer is out of workspace, input[" << plan.inputIndex
                         << "] end=" << tempBufferEnd << ", workspaceSize=" << workspaceSize;
    }

    void* tempBuffer = OffsetWorkspace(workspace, plan.tempBufferOffset);
    plan.tempTensor->UpdateData(tempBuffer);
    inputAddrOverrides[plan.inputIndex] = plan.tempTensor->DataPtr();
    if (plan.tempBufferSize == 0) {
      continue;
    }

    const size_t copyWorkspaceEnd = plan.copyWorkspaceOffset + plan.copyWorkspaceSize;
    if (copyWorkspaceEnd > workspaceSize) {
      RT_GLOG(EXCEPTION) << "DVM V2 contiguous input copy workspace is out of workspace, input[" << plan.inputIndex
                         << "] end=" << copyWorkspaceEnd << ", workspaceSize=" << workspaceSize;
    }
    uint64_t actualCopyWorkspaceSize = 0;
    AclnnExecutor executor("aclnnInplaceCopy");
    executor.GetWorkspaceSize(&actualCopyWorkspaceSize, plan.tempTensor, tensor);
    if (actualCopyWorkspaceSize > plan.copyWorkspaceSize) {
      RT_GLOG(EXCEPTION) << "DVM V2 contiguous input copy workspace grew at launch, input[" << plan.inputIndex
                         << "] planned=" << plan.copyWorkspaceSize << ", actual=" << actualCopyWorkspaceSize;
    }
    executor.Launch(
        OffsetWorkspace(workspace, plan.copyWorkspaceOffset), plan.copyWorkspaceSize, stream, plan.tempTensor, tensor);
  }
  return inputAddrOverrides;
}

OpsErrorCode OpDvmCallV2::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  (void)input;
  CHECK_IF_NULL(workspaceSize);
  if (isDynamic_) {
    UpdateDynamicShapeRefs();
    rawKernel_->Normalize();
  }
  auto* mutableOutput = const_cast<ir::Value*>(output);
  auto addrs = BuildAddressVector(realInputs_, mutableOutput, nullptr, true);
  UpdateRelocAddrs(*relocs_, addrs);
  if (!isSplit_) {
    if (isDynamic_) {
      workspaceSize_ = rawKernel_->PreCodeGen();
      {
        py::gil_scoped_acquire gil;
        CHECK_IF_NULL(kernelObj_);
        kernelObj_->attr("set_workspace_size")(workspaceSize_);
      }
    }
    totalWorkspaceSize_ = PlanContiguousInputs(AlignWorkspaceSize(workspaceSize_));
    *workspaceSize = totalWorkspaceSize_;
    RT_GLOG(INFO) << "DVM V2 workspace size: " << *workspaceSize << ", kernel workspace size: " << workspaceSize_;
    return SUCCESS;
  }
  size_t dvmWorkspaceSize = 0;
  WorkspaceSizeRecorder recorder(&dvmWorkspaceSize);
  rawKernel_->CodeGen(relocs_->data(), relocs_->size(), &recorder);
  {
    py::gil_scoped_acquire gil;
    CHECK_IF_NULL(kernelObj_);
    kernelObj_->attr("set_workspace_size")(dvmWorkspaceSize);
  }
  workspaceSize_ = dvmWorkspaceSize;
  totalWorkspaceSize_ = PlanContiguousInputs(AlignWorkspaceSize(workspaceSize_));
  *workspaceSize = totalWorkspaceSize_;
  RT_GLOG(INFO) << "DVM V2 workspace size: " << *workspaceSize << ", kernel workspace size: " << workspaceSize_;
  return SUCCESS;
}

OpsErrorCode OpDvmCallV2::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  (void)input;
  CHECK_IF_NULL(stream);
  const auto requiredWorkspace = totalWorkspaceSize_;
  if (workspaceSize < requiredWorkspace) {
    RT_GLOG(ERROR) << "DVM V2 workspace is too small, got " << workspaceSize << ", need " << requiredWorkspace;
    return INVALID_PARAM;
  }
  if (requiredWorkspace > 0 && workspace == nullptr) {
    RT_GLOG(EXCEPTION) << "DVM V2 workspace is required but null, need " << requiredWorkspace
                       << ", kernel workspace size=" << workspaceSize_;
  }
  auto inputAddrOverrides = PrepareContiguousInputs(workspace, workspaceSize, stream);
  auto addrs = BuildAddressVector(realInputs_, output, &inputAddrOverrides, true);
  UpdateRelocAddrs(*relocs_, addrs);
  int ret = 0;
  if (isSplit_) {
    ExternalWorkspaceAllocator allocator(workspace);
    rawKernel_->CodeGen(relocs_->data(), relocs_->size(), &allocator);
    ret = rawKernel_->Launch(stream);
  } else if (isDynamic_) {
    ret = rawKernel_->Launch(relocs_->data(), relocs_->size(), workspace, stream);
  } else {
    ret = rawKernel_->Launch(relocs_->data(), relocs_->size(), workspace, stream);
  }
  if (ret != 0) {
    RT_GLOG(ERROR) << "DVM V2 launch failed, ret=" << ret;
    return LAUNCH_OP_FAILED;
  }
  return SUCCESS;
}

} // namespace ops
} // namespace fxrt
