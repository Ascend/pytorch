#ifndef BUILD_LIBTORCH

#include "torch_npu/csrc/inductor/static_launcher/bindings.h"

#include <memory>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/inductor/static_launcher/runtime.h"

namespace py = pybind11;
using torch_npu::inductor::StaticNpuKernel;

namespace {

struct StaticNpuLauncher final {};

} // namespace

void RegisterNPUStaticLauncherBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<StaticNpuKernel, std::shared_ptr<StaticNpuKernel>>(m, "_NPUStaticLoadedKernel");
  py::class_<StaticNpuLauncher>(m, "_StaticNpuLauncher")
      .def_static("_is_supported", &StaticNpuKernel::IsSupported)
      .def_static(
          "_load_kernel",
          &StaticNpuKernel::Load,
          py::arg("binary"),
          py::arg("kernel_name"),
          py::arg("device"),
          py::arg("arg_kinds"),
          py::arg("mix_mode"),
          py::arg("enable_simt"),
          py::arg("shared_mem_dynamic_size"),
          py::arg("is_pure_simt"),
          py::arg("target_support_ffts"),
          py::arg("trailing_pointer_count"))
      .def_static(
          "_launch_kernel",
          [](const std::shared_ptr<StaticNpuKernel>& kernel,
             uint32_t grid0,
             uint32_t grid1,
             uint32_t grid2,
             uint64_t stream,
             const py::sequence& args) {
            TORCH_CHECK(kernel != nullptr, "NPU static launcher kernel is null");
            kernel->Launch(grid0, grid1, grid2, stream, args);
          },
          py::arg("loaded_kernel"),
          py::arg("grid_0"),
          py::arg("grid_1"),
          py::arg("grid_2"),
          py::arg("stream"),
          py::arg("args"))
      .def_static(
          "_unload_kernel",
          [](const std::shared_ptr<StaticNpuKernel>& kernel) {
            TORCH_CHECK(kernel != nullptr, "NPU static launcher kernel is null");
            kernel->Close();
          },
          py::arg("loaded_kernel"));
}

#endif
