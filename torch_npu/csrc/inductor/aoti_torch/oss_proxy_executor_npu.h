#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <nlohmann/json.hpp>
#include <torch_npu/csrc/inductor/aoti_torch/c/shim.h>
#include <torch_npu/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch_npu/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <iostream>
#include <utility>

namespace torch::aot_inductor {

class OSSProxyExecutorNpu : public ProxyExecutor {
 public:
  explicit OSSProxyExecutorNpu(const std::string& json_path, bool is_cpu);

  void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) override;

 private:
  void prefill_stack_with_static_arguments(
      size_t index,
      const at::TypePtr& schema_arg_type,
      const nlohmann::json& serialized_arg,
      OSSOpKernel& op_kernel);

  void get_input_info_from_serialized(
      const std::vector<c10::Argument>& schema_args,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  void get_output_info_from_serialized(
      const std::vector<c10::Argument>& schema_returns,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  std::vector<OSSOpKernel> op_kernels_;
  std::unique_ptr<c10::Device> device_;
};

} // namespace torch::aot_inductor
