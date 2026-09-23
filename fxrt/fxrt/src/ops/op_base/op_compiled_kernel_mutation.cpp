#include "ops/op_base/op_compiled_kernel_mutation.h"

#include <cstddef>
#include <string>
#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"

namespace fxrt {
namespace ops {
namespace {
constexpr size_t kKernelIdxInputIndex = 0;
constexpr size_t kMutatedArgIndicesInputIndex = 1;
constexpr size_t kRealArgInputOffset = 2;
// Owner of the compiled-kernel side table the fx_wrapper fills while it lowers the inductor host graph.
constexpr auto kCompiledKernelModule = "fxrt.compiled_kernel_hop";
constexpr auto kCompiledKernelGetter = "get_compiled_kernel";

py::function ResolveCompiledKernel(int64_t kernelIdx) {
  py::gil_scoped_acquire gil;
  try {
    py::module_ mod = py::module_::import(kCompiledKernelModule);
    py::object kernel = mod.attr(kCompiledKernelGetter)(kernelIdx);
    if (kernel.is_none()) {
      RT_GLOG(EXCEPTION) << "Compiled kernel " << kernelIdx << " is None.";
    }
    return kernel.cast<py::function>();
  } catch (const std::exception& e) {
    RT_GLOG(EXCEPTION) << "Failed to get compiled kernel " << kernelIdx << " from " << kCompiledKernelModule << "."
                       << kCompiledKernelGetter << ": " << e.what();
  }
}

std::vector<size_t> ParseMutatedArgIndices(const ir::Value* value) {
  CHECK_IF_NULL(value);
  if (!value->IsTuple()) {
    RT_GLOG(EXCEPTION) << "compiled_kernel_mutation expects input[" << kMutatedArgIndicesInputIndex
                       << "] to be a tuple of mutated arg indices, got: " << *value;
  }
  const auto& tuple = value->ToTuple();
  std::vector<size_t> indices;
  indices.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); ++i) {
    const auto* elem = (*tuple)[i].get();
    CHECK_IF_NULL(elem);
    if (!elem->IsInt() || elem->ToInt() < 0) {
      RT_GLOG(EXCEPTION) << "compiled_kernel_mutation mutated arg index[" << i
                         << "] must be a non-negative int, got: " << *elem;
    }
    (void)indices.emplace_back(static_cast<size_t>(elem->ToInt()));
  }
  return indices;
}

std::vector<const ir::Value*> FlattenOutputs(const ir::Value* output) {
  CHECK_IF_NULL(output);
  if (output->IsTensor()) {
    return {output};
  }
  if (output->IsTuple()) {
    const auto& tuple = output->ToTuple();
    std::vector<const ir::Value*> outputs;
    outputs.reserve(tuple->Size());
    for (size_t i = 0; i < tuple->Size(); ++i) {
      const auto* elem = (*tuple)[i].get();
      CHECK_IF_NULL(elem);
      if (!elem->IsTensor()) {
        RT_GLOG(EXCEPTION) << "compiled_kernel_mutation output[" << i << "] must be a Tensor, got: " << *elem;
      }
      (void)outputs.emplace_back(elem);
    }
    return outputs;
  }
  RT_GLOG(EXCEPTION) << "compiled_kernel_mutation output must be Tensor or Tuple[Tensor], got: " << *output;
}
} // namespace

OpCompiledKernelMutation::~OpCompiledKernelMutation() {
  // The side table owns the kernel; drop this reference under the GIL so the graph can be destroyed from any thread.
  if (Py_IsInitialized() != 0 && pyFunc_.ptr() != nullptr) {
    py::gil_scoped_acquire gil;
    pyFunc_ = py::function();
  }
}

void OpCompiledKernelMutation::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  SetOpType(OpType::PythonCallOp);
  if (inputs.size() < kRealArgInputOffset) {
    RT_GLOG(EXCEPTION) << "compiled_kernel_mutation expects at least " << kRealArgInputOffset << " inputs, got "
                       << inputs.size();
  }

  const auto* kernelIdxValue = inputs[kKernelIdxInputIndex];
  CHECK_IF_NULL(kernelIdxValue);
  if (!kernelIdxValue->IsInt() || kernelIdxValue->ToInt() < 0) {
    RT_GLOG(EXCEPTION) << "compiled_kernel_mutation expects input[" << kKernelIdxInputIndex
                       << "] to be a non-negative compiled-kernel index, got: " << *kernelIdxValue;
  }
  kernelIdx_ = kernelIdxValue->ToInt();
  auto mutatedArgIndices = ParseMutatedArgIndices(inputs[kMutatedArgIndicesInputIndex]);

  auto outputs = FlattenOutputs(output);
  if (outputs.size() != mutatedArgIndices.size()) {
    RT_GLOG(EXCEPTION) << "compiled_kernel_mutation output count " << outputs.size() << " does not match its "
                       << mutatedArgIndices.size() << " mutated args.";
  }

  const size_t argNum = inputs.size() - kRealArgInputOffset;
  inputs_.assign(inputs.begin() + kRealArgInputOffset, inputs.end());

  // Ref each output to the arg the kernel writes it into, so the two share one buffer.
  refPairs_.clear();
  refPairs_.reserve(mutatedArgIndices.size());
  std::vector<bool> isMutated(argNum, false);
  for (size_t i = 0; i < mutatedArgIndices.size(); ++i) {
    const size_t argIndex = mutatedArgIndices[i];
    if (argIndex >= argNum) {
      RT_GLOG(EXCEPTION) << "compiled_kernel_mutation mutated arg index " << argIndex << " is out of range for "
                         << argNum << " kernel args.";
    }
    if (isMutated[argIndex]) {
      RT_GLOG(EXCEPTION) << "compiled_kernel_mutation got duplicated mutated arg index " << argIndex << ".";
    }
    isMutated[argIndex] = true;
    const auto* arg = inputs_[argIndex];
    CHECK_IF_NULL(arg);
    if (!arg->IsTensor()) {
      RT_GLOG(EXCEPTION) << "compiled_kernel_mutation mutated arg[" << argIndex << "] must be a Tensor, got: " << *arg;
    }
    (void)refPairs_.emplace_back(static_cast<uint32_t>(i), static_cast<uint32_t>(kRealArgInputOffset + argIndex));
  }

  inputTagIndices_.clear();
  inputTagIndices_.reserve(argNum);
  for (const auto* arg : inputs_) {
    (void)inputTagIndices_.emplace_back(static_cast<size_t>(arg->GetTag()));
  }

  moduleName_ = kCompiledKernelModule;
  opName_ = "compiled_kernel_" + std::to_string(kernelIdx_);
  pyFunc_ = ResolveCompiledKernel(kernelIdx_);

  // Reset the zero-copy at::Tensor cache for this initialization.
  firstRun_ = true;
  atTensors_.clear();
  tensorIdx_ = 0;
  RT_GLOG(INFO) << "OpCompiledKernelMutation initialized, kernel=" << kernelIdx_ << ", args=" << argNum
                << ", mutated args=" << refPairs_.size();
}

OpsErrorCode OpCompiledKernelMutation::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  (void)input;
  (void)workspaceSize;
  // A ref'd output stands for the very buffer its input arg holds, so it has to describe that
  // buffer completely. The runtime's generic ref sync only copies strides and storage offset;
  // storage shape and format say how the bytes are laid out, and consumers (aclnn ones especially)
  // read them straight off the tensor they are handed.
  auto outputs = FlattenOutputs(output);
  for (const auto& refPair : refPairs_) {
    const auto& outputTensor = outputs[refPair.first]->ToTensor();
    const auto& argTensor = inputs_[refPair.second - kRealArgInputOffset]->ToTensor();
    CHECK_IF_NULL(outputTensor);
    CHECK_IF_NULL(argTensor);
    outputTensor->SetStorageShape(argTensor->StorageShape());
    outputTensor->SetFormat(argTensor->Format());
  }

  // The kernel runs from here, like every other op that calls out to python: see the class comment.
  // Its buffers are all set by now -- outputs alias their args, and AllocateMemory ran before this.
  if (Py_IsInitialized() == 0) {
    RT_GLOG(EXCEPTION) << "Python interpreter is not initialized.";
    return UNKNOWN_ERROR;
  }
  if (pyFunc_.is_none()) {
    RT_GLOG(EXCEPTION) << "Compiled kernel " << kernelIdx_ << " is not resolved.";
    return UNKNOWN_ERROR;
  }

  py::gil_scoped_acquire gil;
  tensorIdx_ = 0;
  py::tuple pyArgs(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto tagIdx = inputTagIndices_[i];
    if (tagIdx >= kConverterCount) {
      RT_GLOG(EXCEPTION) << "Invalid input tag: " << tagIdx << " at index " << i;
    }
    pyArgs[i] = (this->*kInputConverterTable[tagIdx])(inputs_[i]);
  }

  try {
    pyFunc_(*pyArgs);
  } catch (const std::exception& e) {
    RT_GLOG(EXCEPTION) << "Compiled kernel " << kernelIdx_ << " launch failed: " << e.what();
    return LAUNCH_OP_FAILED;
  }
  firstRun_ = false;
  RT_VLOG(VL_OPS) << "Compiled kernel " << kernelIdx_ << " launched with " << pyArgs.size() << " args.";
  return SUCCESS;
}

} // namespace ops
} // namespace fxrt
