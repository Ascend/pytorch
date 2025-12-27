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

class PyShapeOpStrategy : public torch::aot_inductor::ShapeOpStrategy {
public:
    using torch::aot_inductor::ShapeOpStrategy::ShapeOpStrategy;

    void Transform(
        // 传入NPUShapeHandling的核心配置，供策略使用
        const std::vector<int> &sizes, int min_size, int max_size,
        // 原Transform的输入输出参数
        std::vector<at::Tensor> &inputs, std::vector<int> &indexs, std::vector<std::vector<at::Tensor>> &outputs,
        int dim = 0, double value = 0.0) override
    {
        // for pure virtual function
        PYBIND11_OVERRIDE_PURE(void,
            torch::aot_inductor::ShapeOpStrategy,
            Transform,
            sizes,
            min_size,
            max_size,
            inputs,
            indexs,
            outputs,
            dim,
            value);
    }

    void Recover(
        // 传入NPUShapeHandling的核心配置
        const std::vector<int> &sizes, int min_size, int max_size,
        // 原Recover的输入输出参数
        std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) override
    {
        // for pure virtual function
        PYBIND11_OVERRIDE_PURE(
            void, torch::aot_inductor::ShapeOpStrategy, Recover, sizes, min_size, max_size, inputs, outputs);
    }
};

void THNPShapeHandling_init(PyObject *module)
{
    // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
    // but CI linter and some builds prefer "module".
    auto torch_N_m = py::handle(module).cast<py::module>();

    // ------------------------------ 1. 绑定抽象基类 ShapeOpStrategy（使用PyShapeOpStrategy包装）
    // ------------------------------
    // 关键变更：基类绑定指向PyShapeOpStrategy（包装类），而非原抽象基类，确保多态路由生效
    py::class_<torch::aot_inductor::ShapeOpStrategy,
        PyShapeOpStrategy,
        std::shared_ptr<torch::aot_inductor::ShapeOpStrategy>>(torch_N_m, "_ShapeOpStrategy")
        // 绑定Transform：直接使用基类纯虚函数地址，无需py::pure_virtual（包装类已通过OVERRIDE接管）
        .def(py::init<>())
        .def(
        "transform",
        &torch::aot_inductor::ShapeOpStrategy::Transform,
        "Execute shape transform (PAD/SPLIT)",
        py::arg("sizes"),
        py::arg("min_size"),
        py::arg("max_size"),
        py::arg("inputs"),
        py::arg("indexs"),
        py::arg("outputs"),
        py::arg("dim") = 0,
        py::arg("value") = 0.0
        )
        // 绑定Recover：同理，使用基类纯虚函数地址
        .def(
        "recover",
        &torch::aot_inductor::ShapeOpStrategy::Recover,
        "Recover original shape from transformed tensors",
        py::arg("sizes"),
        py::arg("min_size"),
        py::arg("max_size"),
        py::arg("inputs"),
        py::arg("outputs")
        )
        // 绑定辅助函数FindClosestSize（非纯虚函数，正常绑定）
        .def(
        "find_closest_size",
        &torch::aot_inductor::ShapeOpStrategy::FindClosestSize,
        py::arg("target_size"),
        py::arg("sizes"),
        py::arg("min_size"),
        py::arg("max_size")
        );

    // ------------------------------ 2. 绑定默认策略类 DefaultShapeOp ------------------------------
    py::class_<torch::aot_inductor::DefaultShapeOp,
        torch::aot_inductor::ShapeOpStrategy,
        std::shared_ptr<torch::aot_inductor::DefaultShapeOp>>(torch_N_m, "_DefaultShapeOp")
        .def(py::init<>(), "Default shape op strategy (PAD/SPLIT)");

    // ------------------------------ 3. 绑定核心管理类 NPUShapeHandling ------------------------------
    shared_ptr_class_<torch::aot_inductor::NPUShapeHandling>(torch_N_m, "_NPUShapeHandling")
        .def(py::init<std::vector<int> &>(), py::arg("sizes"))
        .def(py::init([](int min_size, int max_size, std::string &policy_str) {
            torch::aot_inductor::ShapePolicy policy;
            if (policy_str == "times" || policy_str == "TIMES") {
                policy = torch::aot_inductor::ShapePolicy::TIMES;
            } else if (policy_str == "constant" || policy_str == "CONSTANT") {
                policy = torch::aot_inductor::ShapePolicy::CONSTANT;
            } else {
                TORCH_CHECK(false, "Unknown shape policy. Expected `times` or `constant`, got ", policy_str);
            }
            return new torch::aot_inductor::NPUShapeHandling(min_size, max_size, policy);
        }),
             py::arg("min_size"),
             py::arg("max_size"),
             py::arg("policy"))
        // 策略注册接口：使用shared_ptr避免所有权问题
        .def(
        "register_strategy",
        [](torch::aot_inductor::NPUShapeHandling &self,
            std::shared_ptr<torch::aot_inductor::ShapeOpStrategy>
                strategy) { self.RegisterShapeOpStrategy(std::move(strategy)); },
        "Register custom shape op strategy (replace default)",
        py::arg("custom_strategy")
        )
        .def(
        "transform",
        [](torch::aot_inductor::NPUShapeHandling &self,
            std::vector<at::Tensor> &inputs,
            std::vector<int> &indexs,
            int dim,
            double value) {
            std::vector<std::vector<at::Tensor>> outputs;
            self.Transform(inputs, indexs, outputs, dim, value);
            return outputs;
        },
        py::arg("inputs"),
        py::arg("indexs"),
        py::arg("dim") = 0,
        py::arg("value") = 0.0
        )
        .def(
        "recover",
        [](torch::aot_inductor::NPUShapeHandling &self, std::vector<std::vector<at::Tensor>> &inputs) {
            std::vector<at::Tensor> outputs;
            self.Recover(inputs, outputs);
            return outputs;
        },
        py::arg("inputs")
        )
        .def("__repr__", [](const torch::aot_inductor::NPUShapeHandling &self) { return "NPUShapeHandling()"; });
}

#endif
