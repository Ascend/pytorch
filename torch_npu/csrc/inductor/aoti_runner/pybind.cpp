#ifndef BUILD_LIBTORCH
#include <torch/csrc/utils/pybind.h>

#ifdef USE_NPU
#include <torch_npu/csrc/inductor/aoti_runner/pybind.h>
#include <torch_npu/csrc/inductor/aoti_runner/model_container_runner_npu.h>
#endif
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch::inductor {

void initAOTIRunnerBindingsNpu() {
  py::module module = py::module::import("torch._C");
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = py::cast<py::module>(rootModule.attr("_aoti"));

#ifdef USE_NPU
  py::class_<AOTIModelContainerRunnerNpu>(m, "AOTIModelContainerRunnerNpu")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<const std::string&, int, const std::string&, const std::string&>())
      .def("run", &AOTIModelContainerRunnerNpu::run, py::arg("inputs"), py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelContainerRunnerNpu::get_call_spec)
      .def("get_constant_names_to_original_fqns", &AOTIModelContainerRunnerNpu::getConstantNamesToOriginalFQNs)
      .def("get_constant_names_to_dtypes", &AOTIModelContainerRunnerNpu::getConstantNamesToDtypes)
      .def("extract_constants_map", &AOTIModelContainerRunnerNpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerNpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerNpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerNpu::*)(std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerNpu::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def("swap_constant_buffer", &AOTIModelContainerRunnerNpu::swap_constant_buffer)
      .def("free_inactive_constant_buffer", &AOTIModelContainerRunnerNpu::free_inactive_constant_buffer)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerNpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif

  m.def("unsafe_alloc_void_ptrs_from_tensors", [](const std::vector<at::Tensor>& tensors) {
    std::vector<AtenTensorHandle> handles = torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(tensors);
    std::vector<void*> result(
        reinterpret_cast<void**>(handles.data()), reinterpret_cast<void**>(handles.data()) + handles.size());
    return result;
  });
  m.def("unsafe_alloc_void_ptr_from_tensor", [](at::Tensor& tensor) {
    return reinterpret_cast<void*>(torch::aot_inductor::new_tensor_handle(std::move(tensor)));
  });
  m.def("alloc_tensors_by_stealing_from_void_ptrs", [](std::vector<void*>& raw_handles) {
    return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
        reinterpret_cast<AtenTensorHandle*>(raw_handles.data()), raw_handles.size());
  });
  m.def("alloc_tensor_by_stealing_from_void_ptr", [](void* raw_handle) {
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(reinterpret_cast<AtenTensorHandle>(raw_handle));
  });
}
} // namespace torch::inductor
#endif
