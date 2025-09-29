#include <thread>
#include <vector>

#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#include <pybind11/chrono.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#endif

#include "torch_npu/csrc/inductor/aoti_torch/npu_shape_handling.h"
#include "torch_npu/csrc/inductor/aoti_package/shape_handling.h"

#ifndef BUILD_LIBTORCH
template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THNPShapeHandling_init(PyObject* module)
{
    // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
    // but CI linter and some builds prefer "module".
    auto torch_N_m = py::handle(module).cast<py::module>();

    shared_ptr_class_<torch::aot_inductor::NPUShapeHandling>(torch_N_m, "_NPUShapeHandling")
        .def(py::init<std::vector<int>&>(), py::arg("sizes"))
        .def(py::init([](int min_size, int max_size, std::string& policy_str) {
            torch::aot_inductor::ShapePolicy policy;
            if (policy_str == "times" || policy_str == "TIMES") {
                policy =  torch::aot_inductor::ShapePolicy::TIMES;
            } else if (policy_str == "constant" || policy_str == "CONSTANT") {
                policy =  torch::aot_inductor::ShapePolicy::CONSTANT;
            } else {
                TORCH_CHECK(false, "Unknown shape policy. Expected `times` or `constant`, got ",
                    policy_str);
            }
            return new torch::aot_inductor::NPUShapeHandling(min_size, max_size, policy);
        }),
            py::arg("min_size"), py::arg("max_size"), py::arg("policy"))
        .def("transform", [](torch::aot_inductor::NPUShapeHandling& self,
                            std::vector<at::Tensor>& inputs,
                            std::vector<int>& indexs,
                            int dim,
                            double value) {
            std::vector<std::vector<at::Tensor>> outputs;
            self.Transform(inputs, indexs, outputs, dim, value);
            return outputs;
        },
        py::arg("inputs"),
        py::arg("indexs"),
        py::arg("dim") = 0,
        py::arg("value") = 0.0)
        .def("recover", [](torch::aot_inductor::NPUShapeHandling& self,
                          std::vector<std::vector<at::Tensor>>& inputs) {
            std::vector<at::Tensor> outputs;
            self.Recover(inputs, outputs);
            return outputs;
        },
        py::arg("inputs"))
        .def("__repr__", [](const torch::aot_inductor::NPUShapeHandling& self) {
            return "NPUShapeHandling()";
        });
}

#endif
