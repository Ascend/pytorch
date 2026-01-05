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

class PyBSShapeOpStrategy : public torch::aot_inductor::BSShapeOpStrategy {
public:
    using torch::aot_inductor::BSShapeOpStrategy::BSShapeOpStrategy;

    void Transform(std::vector<at::Tensor> &inputs, std::vector<std::vector<at::Tensor>> &outputs) override
    {
        // for pure virtual function
        PYBIND11_OVERRIDE_PURE(
            void,
            torch::aot_inductor::BSShapeOpStrategy,
            Transform,
            inputs,
            outputs
        );
    }

    void Recover(std::vector<std::vector<at::Tensor>> &inputs, std::vector<at::Tensor> &outputs) override
    {
        // for pure virtual function
        PYBIND11_OVERRIDE_PURE(
            void,
            torch::aot_inductor::BSShapeOpStrategy,
            Recover,
            inputs,
            outputs
        );
    }
};

class PySeqShapeOpStrategy : public torch::aot_inductor::SeqShapeOpStrategy {
public:
    using torch::aot_inductor::SeqShapeOpStrategy::SeqShapeOpStrategy;

    void Transform(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs) override
    {
        // for pure virtual function
        PYBIND11_OVERRIDE_PURE(
            void,
            torch::aot_inductor::SeqShapeOpStrategy,
            Transform,
            inputs,
            outputs
        );
    }
};

void THNPShapeHandling_init(PyObject *module)
{
    auto torch_N_m = py::handle(module).cast<py::module>();

    py::enum_<torch::aot_inductor::ShapeType>(torch_N_m, "ShapeType")
        .value("BATCHSIZE", torch::aot_inductor::ShapeType::BATCHSIZE)
        .value("SEQLEN", torch::aot_inductor::ShapeType::SEQLEN)
        .export_values();
    
    py::enum_<torch::aot_inductor::ShapePolicy>(torch_N_m, "ShapePolicy")
        .value("TIMES", torch::aot_inductor::ShapePolicy::TIMES)
        .value("CUSTOM", torch::aot_inductor::ShapePolicy::CUSTOM)
        .export_values();

    shared_ptr_class_<torch::aot_inductor::ShapeOpStrategyBase>(torch_N_m, "_ShapeOpStrategyBase")
        .def(py::init<>())
        .def("find_closest_gear",
             &torch::aot_inductor::ShapeOpStrategyBase::FindClosestGear,
             py::arg("cur_size"),
             py::arg("gears"),
             py::arg("min_gear"),
             py::arg("max_gear")
        )
        .def("pack_op_info",
             &torch::aot_inductor::ShapeOpStrategyBase::EncodeTransformStep,
             py::arg("op"),
             py::arg("index"),
             py::arg("original_size"),
             py::arg("dimension")
        )
        .def("unpack_op_info",
             &torch::aot_inductor::ShapeOpStrategyBase::DecodeTransformStep,
             py::arg("operation"),
             py::arg("op"),
             py::arg("index"),
             py::arg("original_size"),
             py::arg("dimension")
        )
        .def_readwrite("indices", &torch::aot_inductor::ShapeOpStrategyBase::m_indices)
        .def_readwrite("value", &torch::aot_inductor::ShapeOpStrategyBase::m_value)
        .def_readwrite("min_gear", &torch::aot_inductor::ShapeOpStrategyBase::m_min_gear)
        .def_readwrite("max_gear", &torch::aot_inductor::ShapeOpStrategyBase::m_max_gear)
        .def_readwrite("gears", &torch::aot_inductor::ShapeOpStrategyBase::m_gears);

    py::class_<torch::aot_inductor::BSShapeOpStrategy, PyBSShapeOpStrategy,
        torch::aot_inductor::ShapeOpStrategyBase,
        std::shared_ptr<torch::aot_inductor::BSShapeOpStrategy>>(torch_N_m, "_BSShapeOpStrategy")
        .def(py::init<>())
        .def("initialize_core",
             &torch::aot_inductor::BSShapeOpStrategy::InitializeCore,
             py::arg("gears"),
             py::arg("dimension"),
             py::arg("indices"),
             py::arg("value") = 0.0
        )
        .def("transform",
             &torch::aot_inductor::BSShapeOpStrategy::Transform,
             "Execute batch size shape transform (PAD/SPLIT)",
             py::arg("inputs"),
             py::arg("outputs")
        )
        .def("recover",
             &torch::aot_inductor::BSShapeOpStrategy::Recover,
             "Recover original shape from transformed batch size tensors",
             py::arg("inputs"),
             py::arg("outputs")
        )
        .def_readwrite("dimension", &torch::aot_inductor::BSShapeOpStrategy::m_dimension);
    
    py::class_<torch::aot_inductor::SeqShapeOpStrategy, PySeqShapeOpStrategy,
        torch::aot_inductor::ShapeOpStrategyBase,
        std::shared_ptr<torch::aot_inductor::SeqShapeOpStrategy>>(torch_N_m, "_SeqShapeOpStrategy")
        .def(py::init<>())
        .def("initialize_core",
             &torch::aot_inductor::SeqShapeOpStrategy::InitializeCore,
             py::arg("gears"),
             py::arg("dimensions"),
             py::arg("indices"),
             py::arg("value") = 0.0
        )
        .def("transform",
             &torch::aot_inductor::SeqShapeOpStrategy::Transform,
             "Execute sequence shape transform (PAD)",
             py::arg("inputs"),
             py::arg("outputs")
        )
        .def_readwrite("dimensions", &torch::aot_inductor::SeqShapeOpStrategy::m_dimensions);

    py::class_<torch::aot_inductor::DefaultBSShapeOp, torch::aot_inductor::BSShapeOpStrategy,
        std::shared_ptr<torch::aot_inductor::DefaultBSShapeOp>>(torch_N_m, "_DefaultBSShapeOp")
        .def(py::init<>(), "Default batch size shape transform (PAD/SPLIT)");

    py::class_<torch::aot_inductor::DefaultSeqShapeOp, torch::aot_inductor::SeqShapeOpStrategy,
        std::shared_ptr<torch::aot_inductor::DefaultSeqShapeOp>>(torch_N_m, "_DefaultSeqShapeOp")
        .def(py::init<>(), "Default sequence shape transform (PAD)");

    shared_ptr_class_<torch::aot_inductor::NPUShapeHandling>(torch_N_m, "_NPUShapeHandling")
        .def(py::init<>())
        .def("register_batch_size_strategy",
             [](torch::aot_inductor::NPUShapeHandling &self,
                std::shared_ptr<torch::aot_inductor::BSShapeOpStrategy> strategy) {
                    self.RegisterBatchSizeStrategy(std::move(strategy));
             },
             "Register custom batch size shape op strategy (replace default)",
             py::arg("custom_strategy")
        )
        .def("register_sequence_strategy",
             [](torch::aot_inductor::NPUShapeHandling &self,
                std::shared_ptr<torch::aot_inductor::SeqShapeOpStrategy> strategy) {
                    self.RegisterSequenceStrategy(std::move(strategy));
             },
             "Register custom sequence shape op strategy (replace default)",
             py::arg("custom_strategy")
        )
        .def("initialize",
             static_cast<void (torch::aot_inductor::NPUShapeHandling::*)(torch::aot_inductor::ShapeType,
                std::vector<int64_t>&, std::vector<int>&, std::vector<int>&,
                double)>(&torch::aot_inductor::NPUShapeHandling::Initialize),
             py::arg("type"),
             py::arg("gears"),
             py::arg("dimensions"),
             py::arg("indices"),
             py::arg("value") = 0.0
        )
        .def("initialize",
             static_cast<void (torch::aot_inductor::NPUShapeHandling::*)(torch::aot_inductor::ShapeType,
                int64_t, int64_t, torch::aot_inductor::ShapePolicy, std::vector<int>&, std::vector<int>&,
                double)>(&torch::aot_inductor::NPUShapeHandling::Initialize),
             py::arg("type"),
             py::arg("min_size"),
             py::arg("max_size"),
             py::arg("policy"),
             py::arg("dimensions"),
             py::arg("indices"),
             py::arg("value") = 0.0
        )
        .def("transform",
             [](torch::aot_inductor::NPUShapeHandling &self, std::vector<at::Tensor> &inputs) {
                std::vector<std::vector<at::Tensor>> outputs;
                self.Transform(inputs, outputs);
                return outputs;
             },
             py::arg("inputs")
        )
        .def("recover",
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
