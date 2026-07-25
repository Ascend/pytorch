#pragma once
#ifndef BUILD_LIBTORCH
#include <torch/csrc/utils/pybind.h>

#ifdef USE_NPU
#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner_npu.h>
#endif
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch::inductor {

void initAOTIRunnerBindingsNpu(PyObject* module);

} // namespace torch::inductor
#endif