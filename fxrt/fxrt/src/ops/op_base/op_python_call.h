#ifndef __OPS_OP_BASE_OP_PYTHON_CALL_H__
#define __OPS_OP_BASE_OP_PYTHON_CALL_H__

#include <pybind11/pybind11.h>
#include <ATen/core/Tensor.h>
#include <string>
#include <vector>
#include "ops/op_register.h"
#include "hardware/hardware_abstract/device_context.h"

namespace fxrt {
namespace ops {
namespace py = pybind11;

class OpPythonCall : public Operator {
 public:
  OpPythonCall() {
    SetOpType(OpType::PythonCallOp);
  }
  ~OpPythonCall() override = default;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output);

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream);

  bool NeedLaunch() override;

 protected:
  // Input conversion functions for Jump Table
  py::object ConvertNoneToPy(const ir::Value* value);
  py::object ConvertTensorToPy(const ir::Value* value);
  py::object ConvertDoubleToPy(const ir::Value* value);
  py::object ConvertIntToPy(const ir::Value* value);
  py::object ConvertBoolToPy(const ir::Value* value);
  py::object ConvertStringToPy(const ir::Value* value);
  py::object ConvertTupleToPy(const ir::Value* value);

  py::tuple PreprocessInputs(const std::vector<const ir::Value*>& input);
  OpsErrorCode PostprocessOutputs(py::handle result, ir::Value* output);

  std::string opName_;
  std::string moduleName_;
  std::vector<const ir::Value*> inputs_;
  py::function pyFunc_;
  device::DeviceContext* dev_ctx_{nullptr};

  // Cached tensors for zero-copy optimization
  std::vector<at::Tensor> atTensors_;
  std::vector<size_t> inputTagIndices_;
  bool firstRun_{true};
  size_t tensorIdx_{0};

  // Jump Table type definition
  using ConvertFunc = py::object (OpPythonCall::*)(const ir::Value*);
  static const ConvertFunc kInputConverterTable[];
  static constexpr size_t kConverterCount = 8; // None, Tensor, Double, Int, Bool, String, Tuple, Symbol
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_OP_BASE_OP_PYTHON_CALL_H__
