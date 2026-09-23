#include "ops/op_base/op_torch_call.h"
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <functional>

#ifdef ENABLE_TORCH_NPU
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#endif

#include "ops/utils/aten_convert.h"
#include "ops/utils/data_convert.h"

namespace fxrt {
namespace ops {
constexpr size_t kRealInputOffset = 1;

namespace {
bool IsShapeUnchanged(const at::Tensor& atTensor, const std::vector<int64_t>& newShape) {
  const auto& oldShape = atTensor.sizes();
  return newShape.size() == oldShape.size() && std::equal(newShape.begin(), newShape.end(), oldShape.begin());
}

bool IsStridesUnchanged(
    const at::Tensor& atTensor,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides) {
  if (newStrides.empty()) {
    return atTensor.is_contiguous() && atTensor.strides().size() == newShape.size();
  }

  const auto& oldStrides = atTensor.strides();
  return newStrides.size() == oldStrides.size() && std::equal(newStrides.begin(), newStrides.end(), oldStrides.begin());
}

bool IsBasicMetadataUnchanged(
    const at::Tensor& atTensor,
    void* newDataPtr,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides,
    size_t newStorageBytes,
    int64_t newStorageOffset) {
  return newDataPtr == atTensor.data_ptr() && IsShapeUnchanged(atTensor, newShape) &&
      IsStridesUnchanged(atTensor, newShape, newStrides) && newStorageBytes == atTensor.storage().nbytes() &&
      newStorageOffset == atTensor.storage_offset();
}

void UpdateSizesAndStrides(
    at::Tensor& atTensor,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides) {
  auto* tensorImpl = atTensor.unsafeGetTensorImpl();
  if (newStrides.empty()) {
    tensorImpl->set_sizes_contiguous(newShape);
    return;
  }
  tensorImpl->set_sizes_and_strides(newShape, newStrides);
}

void UpdateBasicMetadata(
    at::Tensor& atTensor,
    void* newDataPtr,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides,
    size_t newStorageBytes,
    int64_t newStorageOffset) {
  auto* tensorImpl = atTensor.unsafeGetTensorImpl();
  at::DataPtr dataPtr(newDataPtr, atTensor.device());
  atTensor.storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataPtr));
  atTensor.storage().set_nbytes(newStorageBytes);
  tensorImpl->set_storage_offset(newStorageOffset);
  UpdateSizesAndStrides(atTensor, newShape, newStrides);
}

#ifdef ENABLE_TORCH_NPU
torch_npu::NPUStorageDesc& GetNpuStorageDesc(at::Tensor& atTensor) {
  return static_cast<torch_npu::NPUStorageImpl*>(atTensor.storage().unsafeGetStorageImpl())->npu_desc_;
}

bool IsNpuDescUnchanged(
    at::Tensor& atTensor,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStorageShape,
    int64_t newStorageOffset,
    ir::MemoryFormat newFormat) {
  const auto newNpuFormat = ConvertMemoryFormatToAclFormat(newFormat);
  const auto newTypeMeta = c10::scalarTypeToTypeMeta(atTensor.scalar_type());
  const auto& curStrides = atTensor.strides();
  auto& desc = GetNpuStorageDesc(atTensor);

  if (desc.base_sizes_.size() != newShape.size()) {
    return false;
  }
  if (!std::equal(newShape.begin(), newShape.end(), desc.base_sizes_.begin())) {
    return false;
  }
  if (desc.base_strides_.size() != curStrides.size()) {
    return false;
  }
  if (!std::equal(curStrides.begin(), curStrides.end(), desc.base_strides_.begin())) {
    return false;
  }
  if (desc.storage_sizes_.size() != newStorageShape.size()) {
    return false;
  }
  if (!std::equal(newStorageShape.begin(), newStorageShape.end(), desc.storage_sizes_.begin())) {
    return false;
  }
  return desc.base_offset_ == newStorageOffset && desc.npu_format_ == newNpuFormat &&
      desc.origin_format_ == newNpuFormat && desc.data_type_ == newTypeMeta;
}

void UpdateNpuDesc(
    at::Tensor& atTensor,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides,
    const std::vector<int64_t>& newStorageShape,
    int64_t newStorageOffset,
    ir::MemoryFormat newFormat) {
  const auto newNpuFormat = ConvertMemoryFormatToAclFormat(newFormat);
  const auto newTypeMeta = c10::scalarTypeToTypeMeta(atTensor.scalar_type());
  auto& desc = GetNpuStorageDesc(atTensor);

  desc.base_sizes_.assign(newShape.begin(), newShape.end());
  desc.base_offset_ = newStorageOffset;
  if (newStrides.empty()) {
    const auto& curStrides = atTensor.strides();
    desc.base_strides_.assign(curStrides.begin(), curStrides.end());
  } else {
    desc.base_strides_.assign(newStrides.begin(), newStrides.end());
  }
  desc.storage_sizes_.assign(newStorageShape.begin(), newStorageShape.end());
  desc.npu_format_ = newNpuFormat;
  desc.origin_format_ = newNpuFormat;
  desc.data_type_ = newTypeMeta;
}
#endif
} // namespace

// Jump table for input conversion functions indexed by Tag enum
// Order: None(0), Tensor(1), Double(2), Int(3), Bool(4), String(5), Tuple(6), Symbol(7)
const OpTorchCall::ConvertInputsFunc OpTorchCall::inputConverterTable[] = {
    &OpTorchCall::ConvertNoneInputToStack, // Tag::None = 0
    &OpTorchCall::ConvertTensorInputToStack, // Tag::Tensor = 1
    &OpTorchCall::ConvertDoubleInputToStack, // Tag::Double = 2
    &OpTorchCall::ConvertIntInputToStack, // Tag::Int = 3
    &OpTorchCall::ConvertBoolInputToStack, // Tag::Bool = 4
    &OpTorchCall::ConvertStringInputToStack, // Tag::String = 5
    &OpTorchCall::ConvertTupleInputToStack, // Tag::Tuple = 6
    &OpTorchCall::ConvertIntInputToStack // Tag::Symbol = 7 (treated as Int)
};

// Jump table for tuple conversion functions indexed by Tag enum
const OpTorchCall::ConvertTupleFunc OpTorchCall::tupleConverterTable[] = {
    nullptr, // Tag::None = 0
    &OpTorchCall::ConvertTensorTupleToStack, // Tag::Tensor = 1
    &OpTorchCall::ConvertDoubleTupleToStack, // Tag::Double = 2
    &OpTorchCall::ConvertIntTupleToStack, // Tag::Int = 3
    &OpTorchCall::ConvertBoolTupleToStack, // Tag::Bool = 4
    &OpTorchCall::ConvertStringTupleToStack, // Tag::String = 5
    nullptr, // Tag::Tuple = 6 (nested tuple not supported here)
    &OpTorchCall::ConvertIntTupleToStack // Tag::Symbol = 7
};

void OpTorchCall::UpdateTorchTensor(at::Tensor& atTensor, const ir::TensorPtr& fxrtTensor) {
  CHECK_IF_NULL(fxrtTensor);

  void* newDataPtr = const_cast<void*>(fxrtTensor->DataPtr());
  const auto& newShape = fxrtTensor->Shape();
  const auto& newStrides = fxrtTensor->Strides();
  const auto newStorageBytes = fxrtTensor->GetStorage()->SizeBytes();
  const auto newStorageOffset = fxrtTensor->StorageOffset();
  const auto newStorageShape = fxrtTensor->StorageShape().empty() ? fxrtTensor->Shape() : fxrtTensor->StorageShape();
  RT_VLOG(VL_OPS) << "newStorageShape[" << newStorageShape << "]";

  bool metadataUnchanged =
      IsBasicMetadataUnchanged(atTensor, newDataPtr, newShape, newStrides, newStorageBytes, newStorageOffset);

#ifdef ENABLE_TORCH_NPU
  if (metadataUnchanged && atTensor.device().type() == at::DeviceType::PrivateUse1) {
    metadataUnchanged = IsNpuDescUnchanged(atTensor, newShape, newStorageShape, newStorageOffset, fxrtTensor->Format());
  }
#endif
  if (metadataUnchanged) {
    return;
  }

  UpdateBasicMetadata(
      atTensor, fxrtTensor->GetStorage()->Data(), newShape, newStrides, newStorageBytes, newStorageOffset);

#ifdef ENABLE_TORCH_NPU
  if (atTensor.device().type() == at::DeviceType::PrivateUse1) {
    UpdateNpuDesc(atTensor, newShape, newStrides, newStorageShape, newStorageOffset, fxrtTensor->Format());
  }
#endif
}

// Helper functions for ConvertTupleToStack
void OpTorchCall::ConvertTensorTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  std::vector<at::Tensor> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    if (firstRun_) {
      auto atTensor = ToTorchTensor((*tuple)[i]->ToTensor());
      vec.push_back(atTensor);
      atTensors_.push_back(atTensor);
    } else {
      CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
      UpdateTorchTensor(atTensors_[tensorIdx_], (*tuple)[i]->ToTensor());
      vec.push_back(atTensors_[tensorIdx_]);
    }
    tensorIdx_++;
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertIntTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  std::vector<int64_t> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToInt());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertBoolTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  std::vector<bool> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToBool());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertDoubleTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  std::vector<double> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToDouble());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertStringTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  std::vector<std::string> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToString());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertNoneTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack) {
  CHECK_IF_NULL(tuple);
  size_t size = tuple->Size();
  if (size == 0) {
    torch::jit::push(stack, std::vector<int64_t>{});
    return;
  }

  auto element = (*tuple)[0].get();
  ir::Value::Tag tag = element->GetTag();
  auto tagIdx = static_cast<size_t>(tag);

  // Use jump table instead of unordered_map for O(1) dispatch
  if (tagIdx < kTupleConverterCount && tupleConverterTable[tagIdx] != nullptr) {
    (this->*tupleConverterTable[tagIdx])(tuple, stack);
  } else {
    RT_GLOG(EXCEPTION) << "Unsupported tuple element type: " << *element;
  }
}

// Helper functions for ConvertInputsToStack
void OpTorchCall::ConvertTensorInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  if (firstRun_) {
    auto atTensor = ToTorchTensor(value->ToTensor());
    atTensors_.push_back(atTensor);
    torch::jit::push(stack, atTensor);
  } else {
    // update trensor
    CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
    auto& atTensor = atTensors_[tensorIdx_];
    UpdateTorchTensor(atTensor, value->ToTensor());
    torch::jit::push(stack, atTensor);
  }
  tensorIdx_++;
}

void OpTorchCall::ConvertDoubleInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  torch::jit::push(stack, value->ToDouble());
}

void OpTorchCall::ConvertIntInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  torch::jit::push(stack, value->ToInt());
}

void OpTorchCall::ConvertBoolInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  torch::jit::push(stack, value->ToBool());
}

void OpTorchCall::ConvertStringInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  auto str = value->ToString();
  // Format "device:index" with device in {"cpu", "npu"}, index integer -> c10::Device
  size_t colon = str.find(':');
  if (colon != std::string::npos && colon > 0 && colon + 1 < str.size()) {
    std::string device_name = str.substr(0, colon);
    std::string index_str = str.substr(colon + 1);
    if (device_name == "cpu" || device_name == "npu") {
      bool valid_index = !index_str.empty();
      for (char c : index_str) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
          valid_index = false;
          break;
        }
      }
      if (valid_index) {
        c10::DeviceType device_type = (device_name == "cpu") ? c10::DeviceType::CPU : c10::DeviceType::PrivateUse1;
        c10::DeviceIndex device_index = static_cast<c10::DeviceIndex>(std::stoi(index_str));
        torch::jit::push(stack, c10::Device(device_type, device_index));
        return;
      }
    }
  }
  torch::jit::push(stack, value->ToString());
}

void OpTorchCall::ConvertTupleInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  ConvertTupleToStack(value->ToTuple(), stack);
}

void OpTorchCall::ConvertNoneInputToStack(const ir::Value* value, torch::jit::Stack& stack) {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertInputsToStack(const std::vector<const ir::Value*>& inputs, torch::jit::Stack& stack) {
  if (inputs.size() != cachedInputConverters_.size()) {
    RT_GLOG(EXCEPTION) << "Input size mismatch: Init cached " << cachedInputConverters_.size()
                       << " real inputs, but CalcWorkspace got " << inputs.size();
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    (this->*cachedInputConverters_[i])(inputs[i], stack);
  }
}

void OpTorchCall::ToFxrtTensor(ir::Value* output, torch::jit::IValue&& ivalue) const {
  if (ivalue.isTensor() && output->IsTensor()) {
    auto& tensor = ivalue.toTensor();
    auto& outTensor = output->ToTensor();
    auto data_ptr = tensor.storage().set_data_ptr(std::move(c10::DataPtr())); // return the original data ptr.
    auto deleter = data_ptr.get_deleter();
    auto* data_to_release = data_ptr.release_context();
    auto* data = data_ptr.get();
    outTensor->GetStorage()->SetDataPtrFromAten(data, data_to_release, deleter, tensor.storage().nbytes());
    UpdateTensorFromTorch(outTensor, tensor);
  } else if (ivalue.isList()) {
    auto& tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);
    auto list = ivalue.toList();
    if (list.size() != tuple->Size()) {
      RT_GLOG(EXCEPTION) << "List size not match tuple size";
    }
    for (size_t i = 0; i < list.size(); i++) {
      ToFxrtTensor((*tuple)[i].get(), torch::jit::IValue{list.get(i)});
    }
  } else if (output->IsInt()) {
    *output = ir::Value(ivalue.toInt());
  } else if (output->IsDouble()) {
    *output = ir::Value(ivalue.toDouble());
  } else if (output->IsSymbol()) {
    // If output is symbol, we just ignore it.
    return;
  } else if (output->IsNone()) {
    // If output is none, like the aten.index_put op we just ignore it.
    return;
  } else {
    RT_GLOG(EXCEPTION) << "Output Only Support Tensor or List[Tensor], but got type: "
                       << c10::typeKindToString(ivalue.type()->kind());
  }
}

void OpTorchCall::ConvertStackToOutput(ir::Value* output, torch::jit::Stack&& stack) const {
  if (stack.empty()) {
    return;
  }

  if (stack.size() == 1) {
    ToFxrtTensor(output, std::move(stack[0]));
    return;
  }

  auto& tuple = output->ToTuple();
  if (tuple->Size() != stack.size()) {
    RT_GLOG(EXCEPTION) << "Tuple size not match stack size";
  }
  for (size_t i = 0; i < stack.size(); i++) {
    ToFxrtTensor((*tuple)[i].get(), std::move(stack[i]));
  }
}

bool OpTorchCall::MatchOpSchema(
    const std::vector<const ir::Value*>& inputs,
    const std::shared_ptr<torch::jit::Operator> op,
    std::string* mismatch_reason) const {
  auto fail = [mismatch_reason](const std::string& reason) {
    if (mismatch_reason != nullptr) {
      *mismatch_reason = reason;
    }
    return false;
  };

  auto args = op->schema().arguments();
  // First input is op name
  if (args.size() != inputs.size() - kRealInputOffset) {
    return fail(
        "schema requires " + std::to_string(args.size()) + " args, but got " +
        std::to_string(inputs.size() - kRealInputOffset));
  }

  static const std::unordered_map<c10::TypeKind, std::function<bool(const ir::Value*)>> typeCheckMap = {
      {c10::TypeKind::TensorType, [](const ir::Value* val) { return val->IsTensor(); }},
      {c10::TypeKind::NumberType,
       [](const ir::Value* val) { return val->IsDouble() || val->IsInt() || val->IsBool() || val->IsSymbol(); }},
      {c10::TypeKind::IntType, [](const ir::Value* val) { return val->IsInt() || val->IsSymbol(); }},
      {c10::TypeKind::BoolType, [](const ir::Value* val) { return val->IsBool(); }},
      {c10::TypeKind::FloatType, [](const ir::Value* val) { return val->IsDouble(); }},
      {c10::TypeKind::StringType, [](const ir::Value* val) { return val->IsString(); }},
      {c10::TypeKind::TupleType, [](const ir::Value* val) { return val->IsTuple(); }},
      {c10::TypeKind::ListType, [](const ir::Value* val) { return val->IsTuple(); }},
      {c10::TypeKind::NoneType, [](const ir::Value* val) { return val->IsNone(); }}};

  for (size_t i = 0, j = kRealInputOffset; i < args.size(); ++i, ++j) {
    auto type = args[i].type();
    if (type->kind() == c10::TypeKind::OptionalType) {
      if (inputs[j]->IsNone()) {
        continue;
      }
      type = type->castRaw<c10::OptionalType>()->getElementType();
    }

    if (type->kind() == c10::TypeKind::DeviceObjType) {
      continue;
    }

    auto it = typeCheckMap.find(type->kind());
    if (it == typeCheckMap.end()) {
      return fail("input[" + std::to_string(i) + "] type [" + type->str() + "] not supported for schema matching");
    }
    if (!it->second(inputs[j])) {
      return fail(
          "input[" + std::to_string(i) + "] expects [" + type->str() + "], but got [" +
          TagToString(inputs[j]->GetTag()) + "]");
    }
  }
  return true;
}

std::string OpTorchCall::GetInputTypesExpr(const std::vector<const ir::Value*>& inputs) const {
  std::string expr = "(";
  for (size_t i = kRealInputOffset; i < inputs.size(); ++i) {
    if (i > kRealInputOffset) {
      expr += ", ";
    }
    expr += TagToString(inputs[i]->GetTag());
  }
  expr += ")";
  return expr;
}

std::string OpTorchCall::GetAvailableTorchOps() const {
  auto ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(qualifiedOpName_));
  std::stringstream opsStr;
  for (size_t i = 0; i < ops.size(); ++i) {
    opsStr << " Schema [" << (i + 1) << "]: " << ops[i]->schema() << "\n";
  }
  return opsStr.str();
}

void OpTorchCall::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  RT_VLOG(VL_OPS) << "Start init operator: " << qualifiedOpName_ << ", inputs: " << inputs.size();
  auto ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(qualifiedOpName_));
  std::vector<std::pair<std::string, std::string>> schema_mismatch_reasons;
  for (auto& op : ops) {
    std::string mismatch_reason;
    if (MatchOpSchema(inputs, op, &mismatch_reason)) {
      operation_ = op->getOperation();
      break;
    } else {
      schema_mismatch_reasons.emplace_back(c10::toString(op->schema()), mismatch_reason);
    }
  }
  if (!operation_) {
    std::stringstream error_msg;
    error_msg << "No matching schema found for operator: " << qualifiedOpName_ << "\n"
              << "Input types: " << GetInputTypesExpr(inputs) << "\n"
              << "Tried " << ops.size() << " schemas:\n";
    for (size_t i = 0; i < schema_mismatch_reasons.size(); ++i) {
      error_msg << "  [" << (i + 1) << "] " << schema_mismatch_reasons[i].first << " — "
                << schema_mismatch_reasons[i].second << "\n";
    }
    RT_GLOG(EXCEPTION) << error_msg.str();
  }

  firstRun_ = true;
  atTensors_.clear();
  tensorIdx_ = 0;

  // Cache input converters during Init, SKIP the first input (op name) to match CalcWorkspace input
  cachedInputConverters_.clear();
  cachedInputConverters_.reserve(inputs.size() - kRealInputOffset);

  // Start from kRealInputOffset to skip op name, consistent with how OpCustomCall strips inputs
  for (size_t i = kRealInputOffset; i < inputs.size(); ++i) {
    auto* input = inputs[i];
    auto tag = input->GetTag();
    auto tagIdx = static_cast<size_t>(tag);

    if (tagIdx < kInputConverterCount) {
      cachedInputConverters_.push_back(inputConverterTable[tagIdx]);
    } else {
      RT_GLOG(EXCEPTION) << "Invalid input tag: " << static_cast<int>(tag) << " at index " << i;
    }
  }
}

OpsErrorCode OpTorchCall::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool OpTorchCall::NeedLaunch() {
  return false;
}

OpsErrorCode OpTorchCall::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  torch::jit::Stack stack;
  tensorIdx_ = 0;
  // Inputs process, convert to aten tensor and push to stack.
  // Note: input here is already stripped of op name by OpCustomCall::CalcWorkspace
  ConvertInputsToStack(input, stack);
  operation_(stack);
  // Outputs process. Convert aten tensor to ir::Value.
  ConvertStackToOutput(const_cast<ir::Value*>(output), std::move(stack));
  CheckOutputInputRef(input, output, qualifiedOpName_);
  firstRun_ = false;
  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
