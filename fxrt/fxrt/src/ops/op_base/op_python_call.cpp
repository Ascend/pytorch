#include "ops/op_base/op_python_call.h"
#include <cstddef>
#include <cstring>
#include "ir/value/value.h"
#include "ops/utils/data_convert.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "ops/utils/async.h"

namespace fxrt {
namespace ops {
namespace {
constexpr size_t kInputModuleNameIndex = 0;
constexpr size_t kInputFuncNameIndex = 1;
constexpr size_t kRealInputOffset = 2;
constexpr auto kTorchOpsModule = "torch.ops";
constexpr auto kTorchOpsInnerModule = "torch._ops";

/**
 * @brief Update ATen tensor metadata and data pointer without creating new Python object
 */
void UpdateAtenTensor(at::Tensor& atTensor, const ir::TensorPtr& fxrtTensor) {
  CHECK_IF_NULL(fxrtTensor);

  void* newDataPtr = const_cast<void*>(fxrtTensor->DataPtr());
  void* oldDataPtr = atTensor.data_ptr();
  const auto& newShape = fxrtTensor->Shape();
  const auto& oldShape = atTensor.sizes();

  if (newDataPtr == oldDataPtr && newShape.size() == oldShape.size() &&
      std::equal(newShape.begin(), newShape.end(), oldShape.begin())) {
    return;
  }

  auto tensor_impl = atTensor.unsafeGetTensorImpl();
  at::DataPtr dataPtr(newDataPtr, atTensor.device());
  atTensor.storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataPtr));

  auto strides = fxrtTensor->Strides();
  if (strides.empty()) {
    tensor_impl->set_sizes_contiguous(newShape);
  } else {
    tensor_impl->set_sizes_and_strides(newShape, strides);
  }
}

py::function GetPythonCallable(const std::string& moduleName, const std::string& funcName) {
  py::gil_scoped_acquire gil;
  try {
    if (moduleName.rfind(kTorchOpsInnerModule, 0) != 0) {
      py::module_ mod = py::module_::import(moduleName.c_str());
      py::object func = mod.attr(funcName.c_str());
      if (func.is_none()) {
        RT_GLOG(EXCEPTION) << "attribute '" + funcName + "' is None";
      }
      return func.cast<py::function>();
    }

    std::string subModuleName = moduleName.substr(strlen(kTorchOpsInnerModule));
    if (!subModuleName.empty() && subModuleName.front() == '.') {
      subModuleName.erase(0, 1);
    }

    if (subModuleName.empty()) {
      RT_GLOG(EXCEPTION) << "invalid moduleName: " << moduleName;
      return {};
    }

    py::module_ torchOps = py::module_::import(kTorchOpsModule);
    py::object subMod = torchOps.attr(subModuleName.c_str());
    auto func = subMod.attr(funcName.c_str());
    if (func.is_none()) {
      RT_GLOG(EXCEPTION) << "attribute '" + funcName + "' is None";
    }
    return func.cast<py::function>();
  } catch (const std::exception& e) {
    RT_GLOG(EXCEPTION) << "Failed to get Python callable [" + moduleName + "." << funcName + "]: " << e.what();
    return {};
  }
}

void AttachAtenTensorCopy(
    const at::Tensor& atenTensor,
    ir::TensorPtr& irTensor,
    const std::string& logPrefix,
    device::DeviceContext* devCtx) {
  if (!IsTorchTensorStandardLayout(atenTensor)) {
    RT_GLOG(EXCEPTION) << logPrefix + " output is not in standard layout.";
  }
  std::vector<int64_t> atenShape(atenTensor.sizes().begin(), atenTensor.sizes().end());
  if (atenShape != irTensor->Shape()) {
    RT_GLOG(EXCEPTION) << logPrefix << " shape mismatch, expect " << irTensor->Shape() << ", but got " << atenShape;
  }

  void* src = atenTensor.data_ptr();
  void* dst = irTensor->DataPtr();
  const size_t numBytes = irTensor->GetStorage()->SizeBytes();
  auto launchTask = [dst, src, numBytes, devCtx]() -> int {
    auto stream = devCtx->deviceResManager_->GetCurrentStream();
    if (!devCtx->deviceResManager_->AsyncCopy(dst, src, numBytes, device::CopyType::D2D, stream)) {
      RT_GLOG(EXCEPTION) << "PythonCall device-to-device copy failed.";
      return UNKNOWN_ERROR;
    }
    return SUCCESS;
  };

  const auto& launchOpFunc = ops::OpAsync::GetLaunchOpFunc();
  if (launchOpFunc != nullptr) {
    launchOpFunc(logPrefix, launchTask, false);
  } else {
    (void)launchTask();
  }
}

OpsErrorCode CopyTensorOutput(
    py::handle pyElem,
    ir::Value* irElem,
    const std::string& logPrefix,
    device::DeviceContext* devCtx) {
  at::Tensor aten;
  try {
    aten = pyElem.cast<at::Tensor>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " is not a torch.Tensor";
    return UNKNOWN_ERROR;
  }
  ir::TensorPtr irTensor = const_cast<ir::TensorPtr&>(irElem->ToTensor());
  if (!irTensor) {
    RT_GLOG(EXCEPTION) << logPrefix << " output tensor pointer is null.";
    return UNKNOWN_ERROR;
  }
  AttachAtenTensorCopy(aten, irTensor, logPrefix, devCtx);
  return SUCCESS;
}

OpsErrorCode CopyIntOutput(py::handle pyElem, ir::Value* irElem, const std::string& logPrefix) {
  int64_t v;
  try {
    v = pyElem.cast<int64_t>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " cannot convert to int64";
    return UNKNOWN_ERROR;
  }
  *irElem = ir::Value(v);
  return SUCCESS;
}

OpsErrorCode CopyDoubleOutput(py::handle pyElem, ir::Value* irElem, const std::string& logPrefix) {
  double v;
  try {
    v = pyElem.cast<double>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " cannot convert to double";
    return UNKNOWN_ERROR;
  }
  *irElem = ir::Value(v);
  return SUCCESS;
}

OpsErrorCode CopyBoolOutput(py::handle pyElem, ir::Value* irElem, const std::string& logPrefix) {
  bool v;
  try {
    v = pyElem.cast<bool>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " cannot convert to bool";
    return UNKNOWN_ERROR;
  }
  *irElem = ir::Value(v);
  return SUCCESS;
}

OpsErrorCode CopyStringOutput(py::handle pyElem, ir::Value* irElem, const std::string& logPrefix) {
  std::string v;
  try {
    v = pyElem.cast<std::string>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " cannot convert to string";
    return UNKNOWN_ERROR;
  }
  *irElem = ir::Value(std::move(v));
  return SUCCESS;
}

OpsErrorCode CopyNoneOutput(py::handle pyElem, const std::string& logPrefix) {
  if (!pyElem.is_none()) {
    RT_GLOG(EXCEPTION) << logPrefix << " expects None but got non-None";
    return UNKNOWN_ERROR;
  }
  return SUCCESS;
}

OpsErrorCode CopySymbolOutput(py::handle pyElem, ir::Value* irElem, const std::string& logPrefix) {
  int64_t v;
  try {
    v = pyElem.cast<int64_t>();
  } catch (...) {
    RT_GLOG(EXCEPTION) << logPrefix << " cannot convert to int64 for Symbol output";
    return UNKNOWN_ERROR;
  }
  *irElem = ir::Value(ir::MakeIntrusive<ir::SymbolicConst>(v));
  return SUCCESS;
}

OpsErrorCode CopyPyElemToIrOutput(
    py::handle pyElem,
    ir::Value* irElem,
    const std::string& logPrefix,
    device::DeviceContext* devCtx);

OpsErrorCode CopyTupleOutput(
    py::handle pyElem,
    ir::Value* irElem,
    const std::string& logPrefix,
    device::DeviceContext* devCtx) {
  if (!py::isinstance<py::tuple>(pyElem)) {
    RT_GLOG(EXCEPTION) << logPrefix << " expects Python tuple";
    return UNKNOWN_ERROR;
  }
  py::tuple pyTup = pyElem.cast<py::tuple>();
  const auto& irTup = irElem->ToTuple();
  if (irTup->Size() != pyTup.size()) {
    RT_GLOG(EXCEPTION) << logPrefix << " tuple size mismatch: expect " << irTup->Size() << ", got " << pyTup.size();
    return UNKNOWN_ERROR;
  }
  for (size_t j = 0; j < pyTup.size(); ++j) {
    auto elemLogPrefix = logPrefix + "[" + std::to_string(j) + "]";
    auto ret = CopyPyElemToIrOutput(pyTup[j], (*irTup)[j].get(), elemLogPrefix, devCtx);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

// Helper: copy Python value to IR output element based on expected IR type.
OpsErrorCode CopyPyElemToIrOutput(
    py::handle pyElem,
    ir::Value* irElem,
    const std::string& logPrefix,
    device::DeviceContext* devCtx) {
  switch (irElem->GetTag()) {
    case ir::Value::Tag::Tensor:
      return CopyTensorOutput(pyElem, irElem, logPrefix, devCtx);
    case ir::Value::Tag::Int:
      return CopyIntOutput(pyElem, irElem, logPrefix);
    case ir::Value::Tag::Double:
      return CopyDoubleOutput(pyElem, irElem, logPrefix);
    case ir::Value::Tag::Bool:
      return CopyBoolOutput(pyElem, irElem, logPrefix);
    case ir::Value::Tag::String:
      return CopyStringOutput(pyElem, irElem, logPrefix);
    case ir::Value::Tag::None:
      return CopyNoneOutput(pyElem, logPrefix);
    case ir::Value::Tag::Tuple:
      return CopyTupleOutput(pyElem, irElem, logPrefix, devCtx);
    case ir::Value::Tag::Symbol:
      return CopySymbolOutput(pyElem, irElem, logPrefix);
    default:
      RT_GLOG(EXCEPTION) << logPrefix << " unsupported IR output tag";
      return UNKNOWN_ERROR;
  }
}
} // namespace

// Jump Table definition: None(0), Tensor(1), Double(2), Int(3), Bool(4), String(5), Tuple(6), Symbol(7)
const OpPythonCall::ConvertFunc OpPythonCall::kInputConverterTable[] = {
    &OpPythonCall::ConvertNoneToPy, // Tag::None = 0
    &OpPythonCall::ConvertTensorToPy, // Tag::Tensor = 1
    &OpPythonCall::ConvertDoubleToPy, // Tag::Double = 2
    &OpPythonCall::ConvertIntToPy, // Tag::Int = 3
    &OpPythonCall::ConvertBoolToPy, // Tag::Bool = 4
    &OpPythonCall::ConvertStringToPy, // Tag::String = 5
    &OpPythonCall::ConvertTupleToPy, // Tag::Tuple = 6
    &OpPythonCall::ConvertIntToPy // Tag::Symbol = 7 (treated as Int)
};

void OpPythonCall::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  RT_VLOG(VL_OPS) << "Input size: " << inputs.size();
  SetOpType(OpType::PythonCallOp);
  moduleName_ = inputs[kInputModuleNameIndex]->ToString();
  opName_ = inputs[kInputFuncNameIndex]->ToString();

  auto inputSize = inputs.size() - kRealInputOffset;
  inputs_.resize(inputSize, nullptr);
  inputTagIndices_.reserve(inputSize);

  for (size_t i = kRealInputOffset; i < inputs.size(); i++) {
    inputs_[i - kRealInputOffset] = inputs[i];
    inputTagIndices_.push_back(static_cast<size_t>(inputs[i]->GetTag()));
  }

  pyFunc_ = GetPythonCallable(moduleName_, opName_);
  auto deviceId = fxrt::collective::CollectiveManager::Instance().local_rank_id();
  fxrt::device::DeviceContextKey deviceContextKey = {
      hardware::GetDeviceNameByType(hardware::DeviceType::NPU), deviceId};
  dev_ctx_ = fxrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(dev_ctx_);
  CHECK_IF_NULL(dev_ctx_->deviceResManager_);

  // Reset state for new initialization
  firstRun_ = true;
  atTensors_.clear();
  tensorIdx_ = 0;
}

OpsErrorCode OpPythonCall::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool OpPythonCall::NeedLaunch() {
  return false;
}

py::object OpPythonCall::ConvertNoneToPy(const ir::Value* value) {
  return py::none();
}

py::object OpPythonCall::ConvertTensorToPy(const ir::Value* value) {
  if (firstRun_) {
    auto atTensor = ToTorchTensor(value->ToTensor());
    atTensors_.push_back(atTensor);
    tensorIdx_++;
    return py::cast(atTensor);
  } else {
    CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
    auto& atTensor = atTensors_[tensorIdx_];
    UpdateAtenTensor(atTensor, value->ToTensor());
    tensorIdx_++;
    return py::cast(atTensor);
  }
}

py::object OpPythonCall::ConvertIntToPy(const ir::Value* value) {
  return py::cast(value->ToInt());
}

py::object OpPythonCall::ConvertDoubleToPy(const ir::Value* value) {
  return py::cast(value->ToDouble());
}

py::object OpPythonCall::ConvertBoolToPy(const ir::Value* value) {
  return py::cast(value->ToBool());
}

py::object OpPythonCall::ConvertStringToPy(const ir::Value* value) {
  return py::cast(value->ToString());
}

py::object OpPythonCall::ConvertTupleToPy(const ir::Value* value) {
  const auto& tuple = value->ToTuple();
  py::list pyList(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); ++i) {
    auto* elem = tuple->operator[](i).get();
    auto tagIdx = static_cast<size_t>(elem->GetTag());
    if (tagIdx < kConverterCount) {
      pyList[i] = (this->*kInputConverterTable[tagIdx])(elem);
    } else {
      RT_GLOG(EXCEPTION) << "Invalid tuple element tag: " << static_cast<int>(elem->GetTag());
    }
  }
  return std::move(pyList);
}

OpsErrorCode OpPythonCall::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  if (Py_IsInitialized() == 0) {
    RT_GLOG(EXCEPTION) << "Python interpreter is not initialized.";
    return UNKNOWN_ERROR;
  }

  py::gil_scoped_acquire gilAcquire;
  tensorIdx_ = 0;

  // Use Jump Table for O(1) dispatch based on cached type indices
  py::tuple pyArgs(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto tagIdx = inputTagIndices_[i];
    if (tagIdx < kConverterCount) {
      pyArgs[i] = (this->*kInputConverterTable[tagIdx])(inputs_[i]);
    } else {
      RT_GLOG(EXCEPTION) << "Invalid input tag: " << tagIdx << " at index " << i;
    }
  }

  RT_VLOG(VL_OPS) << "input size: " << pyArgs.size();

  if (pyFunc_.is_none()) {
    RT_GLOG(EXCEPTION) << "Python function object is null, func name: " << opName_;
    return UNKNOWN_ERROR;
  }

  py::object result;
  try {
    result = inputs_.empty() ? pyFunc_() : pyFunc_(*pyArgs);
  } catch (const std::exception& e) {
    RT_GLOG(EXCEPTION) << "Python function call failed: " << e.what();
    return UNKNOWN_ERROR;
  }

  auto ret = PostprocessOutputs(result, const_cast<ir::Value*>(output));
  CheckOutputInputRef(inputs_, output, opName_);
  firstRun_ = false;
  return ret;
}

OpsErrorCode OpPythonCall::PostprocessOutputs(py::handle result, ir::Value* output) {
  if (!output || output->IsNone()) {
    RT_VLOG(VL_OPS) << "PythonCall op " << opName_ << " has no output tensor; ";
    return SUCCESS;
  }

  if (py::isinstance<py::tuple>(result)) {
    if (!output->IsTuple()) {
      RT_GLOG(EXCEPTION) << "PythonCall op " << opName_ << " expects tuple but IR output is not tuple.";
    }
    py::tuple tup = result.cast<py::tuple>();
    const auto& irTup = output->ToTuple();
    if (irTup->Size() != tup.size()) {
      RT_GLOG(EXCEPTION) << "PythonCall op " << opName_ << " tuple size mismatch: expect " << irTup->Size() << ", got "
                         << tup.size();
    }
    for (size_t i = 0; i < tup.size(); ++i) {
      auto ret = CopyPyElemToIrOutput(
          tup[i], (*irTup)[i].get(), "PythonCall op " + opName_ + " tuple[" + std::to_string(i) + "]", dev_ctx_);
      if (ret != SUCCESS) {
        return ret;
      }
    }
    RT_VLOG(VL_OPS) << "PythonCall op " << opName_ << " zero-copy attached " << tup.size()
                    << " elements into output tuple.";
    return SUCCESS;
  }

  // Single output: reuse CopyPyElemToIrOutput to support all types (Tensor, Int, Double, Bool, String, None)
  auto ret = CopyPyElemToIrOutput(result, output, "PythonCall op " + opName_, dev_ctx_);
  if (ret != SUCCESS) {
    return ret;
  }
  RT_VLOG(VL_OPS) << "PythonCall op " << opName_ << " copied single output to IR.";
  return SUCCESS;
}

} // namespace ops
} // namespace fxrt
