// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/inductor/dvm/pybind_api.h"

#include <algorithm>
#include <c10/core/SymFloat.h>
#include <torch/csrc/THP.h>
#include <fstream>
#include <stdexcept>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

template <typename T> using shared_ptr_class_ = py::class_<T, std::shared_ptr<T> >;

void TORCH_NPU_API THDVM_init(PyObject* module)
{
    using namespace dvm;
    auto torch_C_m = py::handle(module).cast<py::module>();
    auto dvm_m = torch_C_m.def_submodule("dvm", "DVM bindings");
    pybind11::enum_<KernelType>(dvm_m, "KernelType")
        .value("kVector", KernelType::kVector)
        .value("kCube", KernelType::kCube)
        .value("kMix", KernelType::kMix)
        .value("kParallel", KernelType::kParallel)
        .value("kSplit", KernelType::kSplit)
        .value("kEager", KernelType::kEager)
        .export_values();
    pybind11::enum_<KernelFlag>(dvm_m, "KernelFlag")
        .value("kDynamic", KernelFlag::kDynamic)
        .value("kUnifyWS", KernelFlag::kUnifyWS)
        .value("kSpeculate", KernelFlag::kSpeculate)
        .export_values();

    shared_ptr_class_<NDObjectPy>(dvm_m, "NDObject");
    shared_ptr_class_<NDSymInt>(dvm_m, "NDSymInt");
    shared_ptr_class_<NDSymFloat>(dvm_m, "NDSymFloat");

    shared_ptr_class_<KernelPy>(dvm_m, "Kernel")
        .def(py::init<const KernelType&, uint32_t>())
        .def("set_kernel_info", &KernelPy::SetKernelInfo, "set_kernel_info")
        .def("get_dtype", &KernelPy::GetDtype, "get dtype")
        .def("load", &KernelPy::Load, "load array")
        .def("view_load", &KernelPy::ViewLoad, "load array")
        .def("store", &KernelPy::Store, "store array", py::arg("input"), py::arg("dtype") = py::none())
        .def("set_store_inplace", &KernelPy::SetStoreInplace, "store inplace")
        .def("sqrt", &KernelPy::Unary<UnaryOpType::kSqrt>, "emit sqrt")
        .def("abs", &KernelPy::Unary<UnaryOpType::kAbs>, "emit abs")
        .def("log", &KernelPy::Unary<UnaryOpType::kLog>, "emit log")
        .def("exp", &KernelPy::Unary<UnaryOpType::kExp>, "emit exp")
        .def("reciprocal", &KernelPy::Unary<UnaryOpType::kReciprocal>, "emit reciprocal")
        .def("isfinite", &KernelPy::Unary<UnaryOpType::kIsFinite>, "emit isfinite")
        .def("logical_not", &KernelPy::Unary<UnaryOpType::kLogicalNot>, "emit logical_not")
        .def("round", &KernelPy::Unary<UnaryOpType::kRound>, "emit round")
        .def("floor", &KernelPy::Unary<UnaryOpType::kFloor>, "emit floor")
        .def("ceil", &KernelPy::Unary<UnaryOpType::kCeil>, "emit ceil")
        .def("trunc", &KernelPy::Unary<UnaryOpType::kTrunc>, "emit trunc")
        .def("cast", &KernelPy::Cast, "emit cast op")
        .def("element_any", &KernelPy::ElementAny, "emit element_any op")
        .def("equal", &KernelPy::Binary<BinaryOpType::kEqual>, "emit equal")
        .def("not_equal", &KernelPy::Binary<BinaryOpType::kNotEqual>, "emit not_equal")
        .def("greater", &KernelPy::Binary<BinaryOpType::kGreater>, "emit greater")
        .def("greater_equal", &KernelPy::Binary<BinaryOpType::kGreaterEqual>, "emit greater_equal")
        .def("less", &KernelPy::Binary<BinaryOpType::kLess>, "emit less")
        .def("less_equal", &KernelPy::Binary<BinaryOpType::kLessEqual>, "emit less_equal")
        .def("add", &KernelPy::Binary<BinaryOpType::kAdd>, "emit add")
        .def("sub", &KernelPy::Binary<BinaryOpType::kSub>, "emit sub")
        .def("mul", &KernelPy::Binary<BinaryOpType::kMul>, "emit mul")
        .def("div", &KernelPy::Binary<BinaryOpType::kDiv>, "emit div")
        .def("pow", &KernelPy::Binary<BinaryOpType::kPow>, "emit pow")
        .def("maximum", &KernelPy::Binary<BinaryOpType::kMaximum>, "emit maximum")
        .def("minimum", &KernelPy::Binary<BinaryOpType::kMinimum>, "emit minimum")
        .def("logical_and", &KernelPy::Binary<BinaryOpType::kLogicalAnd>, "emit logical_and")
        .def("logical_or", &KernelPy::Binary<BinaryOpType::kLogicalOr>, "emit logical_or")
        .def("select", &KernelPy::Select, "emit select op")
        .def("broadcast", &KernelPy::Broadcast, "emit broadcast op")
        .def("broadcast_scalar", &KernelPy::BroadcastScalar, "emit broadcast op")
        .def("reshape", &KernelPy::Reshape, "emit reshape op")
        .def("copy", &KernelPy::Copy, "emit copy op")
        .def("sum", &KernelPy::Reduce<ReduceOpType::kSum>, "emit sum")
        .def("max", &KernelPy::Reduce<ReduceOpType::kMax>, "emit max")
        .def("min", &KernelPy::Reduce<ReduceOpType::kMin>, "emit min")
        .def("one_hot", &KernelPy::OneHot, "emit onehot op")
        .def("matmul", &KernelPy::MatMul, "emit matmul op", py::arg("lhs"), py::arg("rhs"), py::arg("trans_a"),
             py::arg("trans_b"), py::arg("bias") = py::none())
        .def("gmm", &KernelPy::GroupedMatMul, "emit grouped_matmul op", py::arg("lhs"), py::arg("rhs"),
             py::arg("trans_a"), py::arg("trans_b"), py::arg("bias"), py::arg("group_list"), py::arg("group_type"),
             py::arg("group_list_type") = 0)
        .def("p_next", &KernelPy::ParallelNext, "parallel next")
        .def("spec_next", &KernelPy::SpecNext, "spec next")
        .def("das", &KernelPy::DisAssemble, "disassemble code")
        .def("dump", &KernelPy::DumpGraph, "dump graph")
        .def("setup", &KernelPy::Setup, "setup")
        .def("__call__", &KernelPy::Call, "run kernel")
        .def_static("set_deterministic", &KernelPy::SetDeterm, "set deterministic")
        .def_static("set_online_tuning", &KernelPy::SetTuning, "set online tuning");

    pybind11::class_<DynKernelPy, KernelPy, std::shared_ptr<DynKernelPy> >(dvm_m, "DynKernel")
        .def(py::init<const KernelType&, uint32_t>())
        .def("intref", &DynKernelPy::MakeNDSymInt, "setup")
        .def("floatref", &DynKernelPy::MakeNDSymFloat, "setup");

    pybind11::class_<GraphSplitKernelPy, KernelPy, std::shared_ptr<GraphSplitKernelPy> >(dvm_m, "GraphSplitKernel")
        .def(py::init<>());

    pybind11::class_<DynGraphSplitKernelPy, DynKernelPy, std::shared_ptr<DynGraphSplitKernelPy> >(dvm_m,
                                                                                                  "DynGraphSplitKernel")
        .def(py::init<>());
}

namespace dvm {
namespace {
DType TorchDtype2DvmDType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Float:
            return DType::kFloat32;
        case at::ScalarType::Half:
            return DType::kFloat16;
        case at::ScalarType::BFloat16:
            return DType::kBFloat16;
        case at::ScalarType::Int:
            return DType::kInt32;
        case at::ScalarType::Long:
            return DType::kInt64;
        case at::ScalarType::Bool:
            return DType::kBool;
        default:
            return DType::kFloat32;
    }
}

at::ScalarType DvmDType2TorchDtype(DType dtype)
{
    switch (dtype) {
        case DType::kFloat32:
            return at::ScalarType::Float;
        case DType::kFloat16:
            return at::ScalarType::Half;
        case DType::kBFloat16:
            return at::ScalarType::BFloat16;
        case DType::kInt32:
            return at::ScalarType::Int;
        case DType::kInt64:
            return at::ScalarType::Long;
        case DType::kBool:
            return at::ScalarType::Bool;
        default:
            throw std::runtime_error("Unsupported Dtype conversion");
    }
}

} // namespace

KernelPy::KernelPy(const KernelType& kernel_type, uint32_t flags)
{
    kernel_.Reset(kernel_type, flags);
}

KernelPy::~KernelPy()
{
    for (auto ref : shapes_) {
        delete ref;
    }
}

void KernelPy::SetDeterm(bool enable)
{
    auto& conf = Config::Instance();
    if (enable) {
        conf.SetDeterm();
    } else {
        conf.UnsetDeterm();
    }
}

void KernelPy::SetTuning(bool enable)
{
    auto& conf = Config::Instance();
    if (enable) {
        conf.SetOnlineTuner().SetLazyTuner();
    } else {
        conf.UnsetOnlineTuner().UnsetLazyTuner();
    }
}

DynKernelPy::~DynKernelPy()
{
    for (auto ref : dyn_load_shapes_) {
        delete ref;
    }
}

at::ScalarType KernelPy::GetDtype(py::object op)
{
    auto obj = op.cast<NDOpPyPtr>()->Get();
    return DvmDType2TorchDtype(kernel_.GetDType(obj));
}

ShapeRef* KernelPy::SymIntArraytoShapeRef(py::object shape)
{
    auto shape_array = shape.cast<py::sequence>();
    auto& ref = shapes_.emplace_back(new ShapeWithRef(shape_array.size()));
    for (size_t i = 0; i < ref->size; ++i) {
        ref->shape_data[i] = shape_array[i].cast<int64_t>();
    }
    return ref;
}

ShapeRef* KernelPy::SymIntArraytoShapeRef(at::IntArrayRef shape_array)
{
    auto& ref = shapes_.emplace_back(new ShapeWithRef(shape_array.size()));
    for (size_t i = 0; i < ref->size; ++i) {
        ref->shape_data[i] = shape_array[i];
    }
    return ref;
}

DynKernelPy::LoadShapeRef* DynKernelPy::GetDynLoadShapeRef(size_t dim_size)
{
    static int64_t init_dyn_data[ShapeWithRef::MAX_SIZE] = {-1};
    auto ref = new LoadShapeRef();
    ref->shape.data = init_dyn_data;
    ref->shape.size = dim_size;
    ref->stride.data = nullptr;
    ref->stride.size = 0;
    dyn_load_shapes_.push_back(ref);
    return ref;
}

template <UnaryOpType op_type> py::object KernelPy::Unary(py::object input)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.Unary(op_type, in_obj);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Cast(py::object input, at::ScalarType dtype)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.Cast(in_obj, TorchDtype2DvmDType(dtype));
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Select(py::object cond, py::object lhs, py::object rhs)
{
    auto input1 = lhs.cast<NDOpPyPtr>()->Get();
    auto input2 = rhs.cast<NDOpPyPtr>()->Get();
    auto input0 = cond.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.Select(input0, input1, input2);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

template <ReduceOpType op_type> py::object KernelPy::Reduce(py::object input, py::object dims, bool keepdims)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto dims_ref = SymIntArraytoShapeRef(dims);
    auto op = kernel_.Reduce(op_type, in_obj, dims_ref, keepdims);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

template <BinaryOpType op_type> py::object KernelPy::Binary(py::object lhs, py::object rhs)
{
    NDObject* op;
    if (py::isinstance<py::int_>(lhs)) {
        op = kernel_.Binary(op_type, lhs.cast<int>(), rhs.cast<NDOpPyPtr>()->Get());
    } else if (py::isinstance<py::float_>(lhs)) {
        op = kernel_.Binary(op_type, lhs.cast<float>(), rhs.cast<NDOpPyPtr>()->Get());
    } else if (py::isinstance<py::int_>(rhs)) {
        op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<int>());
    } else if (py::isinstance<py::float_>(rhs)) {
        op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), rhs.cast<float>());
    } else if (py::isinstance<NDSymInt>(lhs)) {
        op = kernel_.Binary(op_type, &(lhs.cast<NDSymIntPtr>()->data_), rhs.cast<NDOpPyPtr>()->Get());
    } else if (py::isinstance<NDSymFloat>(lhs)) {
        op = kernel_.Binary(op_type, &(lhs.cast<NDSymFloatPtr>()->data_), rhs.cast<NDOpPyPtr>()->Get());
    } else if (py::isinstance<NDSymInt>(rhs)) {
        op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), &(rhs.cast<NDSymIntPtr>()->data_));
    } else if (py::isinstance<NDSymFloat>(rhs)) {
        op = kernel_.Binary(op_type, lhs.cast<NDOpPyPtr>()->Get(), &(rhs.cast<NDSymFloatPtr>()->data_));
    } else {
        auto input1 = lhs.cast<NDOpPyPtr>()->Get();
        auto input2 = rhs.cast<NDOpPyPtr>()->Get();
        op = kernel_.Binary(op_type, input1, input2);
    }
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Broadcast(py::object input, py::object shape)
{
    auto shape_ref = SymIntArraytoShapeRef(shape);
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.Broadcast(in_obj, shape_ref);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::BroadcastScalar(py::object scalar, py::object shape, at::ScalarType dtype)
{
    auto shape_ref = SymIntArraytoShapeRef(shape);
    auto type_id = TorchDtype2DvmDType(dtype);
    NDObject* op;
    if (py::isinstance<py::int_>(scalar)) {
        op = kernel_.Broadcast(scalar.cast<int>(), shape_ref, type_id);
    } else if (py::isinstance<py::float_>(scalar)) {
        op = kernel_.Broadcast(scalar.cast<float>(), shape_ref, type_id);
    } else if (py::isinstance<NDSymInt>(scalar)) {
        op = kernel_.Broadcast(&(scalar.cast<NDSymIntPtr>()->data_), shape_ref, type_id);
    } else {
        op = kernel_.Broadcast(&(scalar.cast<NDSymFloatPtr>()->data_), shape_ref, type_id);
    }
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Reshape(py::object input, py::object shape)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto shape_ref = SymIntArraytoShapeRef(shape);
    auto op = kernel_.Reshape(in_obj, shape_ref);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Copy(py::object input)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.Copy(in_obj);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::OneHot(py::object indices, int depth, int axis, c10::Scalar on_value, c10::Scalar off_value,
                            at::ScalarType dtype)
{
    auto indices_obj = indices.cast<NDOpPyPtr>()->Get();
    auto depth_ref = new ShapeWithRef(1);
    depth_ref->shape_data[0] = depth;
    shapes_.push_back(depth_ref);
    auto type_id = TorchDtype2DvmDType(dtype);
    NDObject* op;
    if (on_value.isIntegral(true)) {
        op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toInt(), off_value.toInt());
    } else {
        op = kernel_.OneHot(indices_obj, depth_ref, axis, on_value.toFloat(), off_value.toFloat());
    }
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Load(at::IntArrayRef shape, at::ScalarType dtype)
{
    ShapeRef* shape_ref = SymIntArraytoShapeRef(shape);
    auto op = kernel_.Load(nullptr, shape_ref, TorchDtype2DvmDType(dtype));
    loads_.emplace_back(op);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::ViewLoad(at::IntArrayRef shape, at::IntArrayRef stride, at::ScalarType dtype)
{
    ShapeRef* shape_ref = SymIntArraytoShapeRef(shape);
    ShapeRef* stride_ref = SymIntArraytoShapeRef(stride);
    const int64_t* offset_ptr = nullptr;
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, offset_ptr, TorchDtype2DvmDType(dtype));
    loads_.emplace_back(op);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object DynKernelPy::Load(at::IntArrayRef shape, at::ScalarType dtype)
{
    auto ref = GetDynLoadShapeRef(shape.size());
    ShapeRef* shape_ref = &ref->shape;
    auto op = kernel_.Load(nullptr, shape_ref, TorchDtype2DvmDType(dtype));
    loads_.emplace_back(op);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object DynKernelPy::ViewLoad(at::IntArrayRef shape, at::IntArrayRef stride, at::ScalarType dtype)
{
    auto ref = GetDynLoadShapeRef(shape.size());
    ref->stride = ref->shape;
    ShapeRef* shape_ref = &ref->shape;
    ShapeRef* stride_ref = &ref->stride;
    const int64_t* offset_ptr = nullptr;
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, offset_ptr, TorchDtype2DvmDType(dtype));
    loads_.emplace_back(op);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::Store(py::object obj, py::object dtype)
{
    auto in_obj = obj.cast<NDOpPyPtr>()->Get();
    if (!dtype.is_none()) {
        auto store_type = TorchDtype2DvmDType(dtype.cast<at::ScalarType>());
        if (store_type != kernel_.GetDType(in_obj)) {
            in_obj = kernel_.Cast(in_obj, store_type);
        }
    }
    auto op = kernel_.Store(nullptr, in_obj);
    return py::cast(std::make_shared<NDObjectPy>(stores_.emplace_back(op)));
}

void KernelPy::SetStoreInplace(py::object obj)
{
    auto store = obj.cast<NDOpPyPtr>()->Get();
    kernel_.SetStoreInplace(store);
}

py::object KernelPy::ElementAny(py::object input)
{
    auto in_obj = input.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.ElemAny(in_obj);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias)
{
    auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
    auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
    auto op =
        kernel_.MatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get());
    return py::cast(std::make_shared<NDObjectPy>(op));
}

py::object KernelPy::GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                                   py::object group_list, int64_t group_type, int64_t group_list_type)
{
    auto lhs_obj = lhs.cast<NDOpPyPtr>()->Get();
    auto rhs_obj = rhs.cast<NDOpPyPtr>()->Get();
    NDObject* bias_obj = bias.is_none() ? nullptr : bias.cast<NDOpPyPtr>()->Get();
    NDObject* group_list_obj = group_list.is_none() ? nullptr : group_list.cast<NDOpPyPtr>()->Get();
    auto op = kernel_.GroupedMatMul(lhs_obj, rhs_obj, trans_a, trans_b, bias_obj, group_list_obj, GroupType(group_type),
                                    (GroupListType)group_list_type);
    return py::cast(std::make_shared<NDObjectPy>(op));
}

void KernelPy::ParallelNext() { kernel_.ParallelNext(); }
void KernelPy::SpecNext() { kernel_.SpecNext(); }

void KernelPy::Setup()
{
    SetupRelocs();
    ws_size_ = kernel_.CodeGen();
}

std::string KernelPy::DisAssemble() { return kernel_.Das(); }

std::string KernelPy::DumpGraph() { return kernel_.Dump(); }

void KernelPy::SetupRelocs()
{
    relocs_.clear();
    relocs_.reserve(loads_.size() + stores_.size());
    for (auto op : loads_) {
        relocs_.emplace_back(op, nullptr);
    }
    for (auto op : stores_) {
        relocs_.emplace_back(op, nullptr);
    }
}

py::object KernelPy::Call(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    TORCH_CHECK(args.size() == num_inputs + num_outputs);
    std::vector<at::Tensor> tensor_list;
    std::vector<std::pair<at::Tensor, at::Tensor> > out_refs;
    auto addr = std::make_shared<std::vector<void*> >();
    tensor_list.reserve(args.size());
    out_refs.reserve(num_outputs);
    addr->resize(num_inputs + num_outputs);
    for (size_t i = 0; i < args.size(); i++) {
        auto tensor = args[i].cast<at::Tensor>();
        if (!contiguity_flags_[i]) {
            if (i < num_inputs) {
                tensor = tensor.contiguous();
            } else {
                TORCH_CHECK(!tensor.is_contiguous());
                tensor = out_refs.emplace_back(tensor, at::empty_like(tensor)).second;
            }
        }
        tensor_list.emplace_back(tensor);
        (*addr)[i] = tensor.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto dvm_call = [this, addr, stream]() { return Launch(addr->data(), stream); };

    at_npu::native::OpCommand::RunOpApiV2(op_name_, dvm_call);

    for (auto [ori_tensor, cur_tensor] : out_refs) {
        ori_tensor.copy_(cur_tensor);
    }
    return py::none();
}

py::object KernelPy::CreateOutputs(const at::TensorOptions& options, void** addr)
{
    auto create_output = [this, options](NDObject* store) -> at::Tensor {
        auto shape_ref = kernel_.GetShape(store);
        c10::IntArrayRef shape = c10::IntArrayRef(shape_ref->data, shape_ref->size);
        at::ScalarType dtype = DvmDType2TorchDtype(kernel_.GetDType(store));
        return at_npu::native::OpPreparation::apply_tensor_without_format(shape, options.dtype(dtype));
    };
    if (stores_.size() == 1) {
        at::Tensor tensor = create_output(stores_.front());
        *addr = tensor.data_ptr();
        return py::cast(tensor);
    }
    py::tuple tuple_ret(stores_.size());
    for (size_t i = 0; i < stores_.size(); ++i) {
        at::Tensor tensor = create_output(stores_[i]);
        *addr = tensor.data_ptr();
        addr++;
        tuple_ret[i] = py::cast(tensor);
    }
    return tuple_ret;
}

void GraphSplitKernelPy::Setup()
{
    SetupRelocs();
    kernel_.Infer();
}

void DynGraphSplitKernelPy::Setup()
{
    SetupRelocs();
}

int GraphSplitBase::Launch(Kernel& kernel, void** addr, aclrtStream stream, std::vector<RelocEntry>& relocs)
{
    stream_ = stream;
    for (size_t i = 0; i < relocs.size(); ++i) {
        relocs[i].addr = addr[i];
    }
    kernel.CodeGen(relocs.data(), relocs.size(), this);
    int ret = kernel.Launch(stream);
    ws_.reset();
    return ret;
}

py::object GraphSplitKernelPy::Call(py::args inputs)
{
    TORCH_CHECK(inputs.size() == loads_.size());
    auto addr = std::make_shared<std::vector<void*> >();
    addr->resize(relocs_.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        auto tensor = inputs[i].cast<at::Tensor>();
        (*addr)[i] = tensor.data_ptr();
    }
    auto options = inputs[0].cast<at::Tensor>().options();
    auto ret = CreateOutputs(options, addr->data() + loads_.size());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch_call = [this, stream, addr]() -> int {
        return GraphSplitBase::Launch(kernel_, addr->data(), stream, relocs_);
    };
    at_npu::native::OpCommand::RunOpApiV2(op_name_, launch_call);
    return ret;
}

py::object DynGraphSplitKernelPy::Call(py::args inputs)
{
    while (entry_cnt_ != exit_cnt_) {
        sleep(0);
    }
    entry_cnt_++;
    auto addr = std::make_shared<std::vector<void*> >();
    addr->resize(relocs_.size());
    at::TensorOptions options;
    size_t input_index = 0;
    size_t sym_int_index = 0;
    size_t sym_float_index = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        if (THPVariable_Check(inputs[i].ptr())) {
            auto tensor = inputs[i].cast<at::Tensor>();
            (*addr)[input_index] = tensor.data_ptr();
            auto ref = dyn_load_shapes_[input_index++];
            if (ref->stride.data) {
                ref->stride.data = tensor.strides().data();
            }
            ref->shape.data = tensor.sizes().data();
            options = tensor.options();
        } else if (py::isinstance<c10::SymFloat>(inputs[i])) {
            sym_float_input_[sym_float_index++]->data_ =
                static_cast<float>(inputs[i].cast<c10::SymFloat>().expect_float());
        } else {
            sym_input_[sym_int_index++]->data_ = static_cast<int32_t>(inputs[i].cast<c10::SymInt>().expect_int());
        }
    }
    for (size_t i = 0; i < sym_shape_.size(); i++) {
        for (size_t j = 0; j < shapes_[i]->size; j++) {
            shapes_[i]->shape_data[j] = static_cast<int64_t>(sym_shape_[i][j]->data_.i32);
        }
    }
    kernel_.Infer();
    auto ret = CreateOutputs(options, addr->data() + loads_.size());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch_call = [this, stream, addr]() -> int {
        auto ret = GraphSplitBase::Launch(kernel_, addr->data(), stream, relocs_);
        exit_cnt_++;
        return ret;
    };
    at_npu::native::OpCommand::RunOpApiV2(op_name_, launch_call);
    return ret;
}

int KernelPy::Launch(void** addr, aclrtStream stream)
{
    void* workspace_ptr = nullptr;
    at::Tensor workspace_tensor;
    if (ws_size_ != 0) {
        workspace_tensor = at_npu::native::allocate_workspace(ws_size_, stream);
        workspace_ptr = const_cast<void*>(workspace_tensor.storage().data());
    }
    for (size_t i = 0; i < relocs_.size(); ++i) {
        relocs_[i].addr = addr[i];
    }
    return kernel_.Launch(relocs_.data(), relocs_.size(), workspace_ptr, stream);
}

ShapeRef* DynKernelPy::SymIntArraytoShapeRef(py::object shape)
{
    auto shape_array = shape.cast<py::sequence>();
    auto& ref = shapes_.emplace_back(new ShapeWithRef(shape_array.size()));
    auto& sym_shape = sym_shape_.emplace_back();
    for (size_t i = 0; i < ref->size; ++i) {
        ref->shape_data[i] = -1;
        if (py::isinstance<py::int_>(shape_array[i])) {
            auto sym_ptr = const_input_.emplace_back(
                std::make_shared<NDSymInt>(static_cast<int32_t>(shape_array[i].cast<int64_t>())));
            sym_shape.emplace_back(sym_ptr);
        } else {
            sym_shape.emplace_back(shape_array[i].cast<NDSymIntPtr>());
        }
    }
    return ref;
}

void DynKernelPy::Setup()
{
    SetupRelocs();
}

py::object DynKernelPy::Call(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    const auto num_sym = sym_input_.size() + sym_float_input_.size();
    const auto num_sym_float = sym_float_input_.size();
    TORCH_CHECK(args.size() == num_inputs + num_outputs + num_sym);
    std::vector<at::Tensor> tensor_list;
    std::vector<std::pair<at::Tensor, at::Tensor> > out_refs;

    auto info = std::make_shared<DynamicInfo>();
    info->shape.reserve(num_inputs);
    info->strides.reserve(num_inputs);
    info->symint.reserve(sym_input_.size());
    info->symfloat.reserve(num_sym_float);
    info->addr.reserve(num_inputs + num_outputs);
    tensor_list.reserve(num_inputs + num_outputs);
    out_refs.reserve(num_outputs);
    for (size_t i = 0; i < args.size(); i++) {
        if (THPVariable_Check(args[i].ptr())) {
            auto tensor = args[i].cast<at::Tensor>();
            if (!contiguity_flags_[i]) {
                if (i < num_inputs + num_sym) {
                    tensor = tensor.contiguous();
                } else {
                    TORCH_CHECK(!tensor.is_contiguous());
                    tensor = out_refs.emplace_back(tensor, at::empty_like(tensor)).second;
                }
            }
            if (i < num_inputs + num_sym) {
                info->shape.emplace_back(tensor.sizes().vec());
                info->strides.emplace_back(tensor.strides().vec());
            }
            tensor_list.emplace_back(tensor);
            info->addr.emplace_back(tensor.data_ptr());
        } else if (py::isinstance<c10::SymFloat>(args[i])) {
            info->symfloat.emplace_back(static_cast<float>(args[i].cast<c10::SymFloat>().expect_float()));
        } else {
            info->symint.emplace_back(static_cast<int32_t>(args[i].cast<c10::SymInt>().expect_int()));
        }
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    auto dvm_call = [this, info, stream]() {
        for (size_t i = 0; i < loads_.size(); i++) {
            auto& ref = dyn_load_shapes_[i];
            if (ref->stride.data) {
                ref->stride.data = info->strides[i].data();
            }
            ref->shape.data = info->shape[i].data();
        }
        for (size_t i = 0; i < sym_input_.size(); i++) {
            sym_input_[i]->data_ = info->symint[i];
        }
        for (size_t i = 0; i < sym_float_input_.size(); i++) {
            sym_float_input_[i]->data_ = info->symfloat[i];
        }
        for (size_t i = 0; i < sym_shape_.size(); i++) {
            for (size_t j = 0; j < shapes_[i]->size; j++) {
                shapes_[i]->shape_data[j] = static_cast<int64_t>(sym_shape_[i][j]->data_.i32);
            }
        }
        ws_size_ = kernel_.CodeGen();
        return Launch(info->addr.data(), stream);
    };

    at_npu::native::OpCommand::RunOpApiV2(op_name_, dvm_call);

    for (auto [ori_tensor, cur_tensor] : out_refs) {
        ori_tensor.copy_(cur_tensor);
    }
    return py::none();
}
} // namespace dvm
#endif // BUILD_LIBTORCH
