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
    RegDvmPy(dvm_m);
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

    pybind11::class_<TorchKernelPy, KernelPy, std::shared_ptr<TorchKernelPy> >(dvm_m, "TorchKernel")
        .def(py::init<const KernelType&, uint32_t>())
        .def("set_kernel_info", &TorchKernelPy::SetKernelInfo, "set_kernel_info")
        .def("setup", &TorchKernelPy::Setup, "setup")
        .def("__call__", &TorchKernelPy::Call, "run kernel");

    pybind11::class_<DynKernelPy, TorchKernelPy, std::shared_ptr<DynKernelPy> >(dvm_m, "DynKernel")
        .def(py::init<const KernelType&, uint32_t>())
        .def("scalar", &DynKernelPy::MakeScalar, "setup", py::arg("dtype") = DataTypePy(kDataTypeEnd));

    pybind11::class_<GraphSplitKernelPy, TorchKernelPy, std::shared_ptr<GraphSplitKernelPy> >(dvm_m, "GraphSplitKernel")
        .def(py::init<>());

    pybind11::class_<DynGraphSplitKernelPy, DynKernelPy, std::shared_ptr<DynGraphSplitKernelPy> >(dvm_m,
                                                                                                  "DynGraphSplitKernel")
        .def(py::init<>());
}

namespace dvm {
namespace {
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

TorchKernelPy::TorchKernelPy(const KernelType& kernel_type, uint32_t flags) { kernel_.Reset(kernel_type, flags); }

TorchKernelPy::~TorchKernelPy()
{
    for (auto ref : shapes_) {
        delete ref;
    }
}

void TorchKernelPy::SetDeterm(bool enable)
{
    auto& conf = Config::Instance();
    if (enable) {
        conf.SetDeterm();
    } else {
        conf.UnsetDeterm();
    }
}

void TorchKernelPy::SetTuning(bool enable)
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

IntArrayRef* TorchKernelPy::GetShapeRef(py::object shape) { return SymIntArraytoShapeRef(shape); }

ShapeRef* TorchKernelPy::SymIntArraytoShapeRef(py::object shape)
{
    auto shape_array = shape.cast<py::sequence>();
    auto& ref = shapes_.emplace_back(new ShapeWithRef(shape_array.size()));
    for (size_t i = 0; i < ref->size; ++i) {
        ref->shape_data[i] = shape_array[i].cast<int64_t>();
    }
    return ref;
}

ShapeRef* TorchKernelPy::SymIntArraytoShapeRef(at::IntArrayRef shape_array)
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

py::object TorchKernelPy::Load(py::object shape, DataTypePy type)
{
    ShapeRef* shape_ref = SymIntArraytoShapeRef(shape);
    auto op = kernel_.Load(nullptr, shape_ref, type);
    loads_.emplace_back(op);
    return ObjToPy(op);
}

py::object TorchKernelPy::ViewLoad(py::object shape, py::object stride, int64_t offset, DataTypePy type)
{
    ShapeRef* shape_ref = SymIntArraytoShapeRef(shape);
    ShapeRef* stride_ref = SymIntArraytoShapeRef(stride);
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, nullptr, type);
    loads_.emplace_back(op);
    return ObjToPy(op);
}

py::object DynKernelPy::Load(py::object shape, DataTypePy type)
{
    auto shape_seq = shape.cast<py::sequence>();
    auto ref = GetDynLoadShapeRef(shape_seq.size());
    ShapeRef* shape_ref = &ref->shape;
    auto op = kernel_.Load(nullptr, shape_ref, type);
    loads_.emplace_back(op);
    return ObjToPy(op);
}

py::object DynKernelPy::ViewLoad(py::object shape, py::object stride, int64_t offset, DataTypePy type)
{
    auto shape_seq = shape.cast<py::sequence>();
    auto ref = GetDynLoadShapeRef(shape_seq.size());
    ref->stride = ref->shape;
    ShapeRef* shape_ref = &ref->shape;
    ShapeRef* stride_ref = &ref->stride;
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, nullptr, type);
    loads_.emplace_back(op);
    return ObjToPy(op);
}

py::object TorchKernelPy::Store(py::object obj, DataTypePy type)
{
    auto in_obj = PyToObj(obj);
    if (type != kDataTypeEnd) {
        in_obj = kernel_.Cast(in_obj, type);
    }
    auto op = kernel_.Store(nullptr, in_obj);
    return ObjToPy(stores_.emplace_back(op));
}

void TorchKernelPy::Setup()
{
    SetupRelocs();
    ws_size_ = kernel_.CodeGen();
}

void TorchKernelPy::SetupRelocs()
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

py::object TorchKernelPy::Call(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    TORCH_CHECK(args.size() == num_inputs + num_outputs, "DVM kernel call expects ", num_inputs + num_outputs,
                " tensors, got ", args.size());
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
                TORCH_CHECK(!tensor.is_contiguous(), "Expected non-contiguous output tensor at index ", i);
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

py::object TorchKernelPy::CreateOutputs(const at::TensorOptions& options, void** addr)
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

void DynGraphSplitKernelPy::Setup() { SetupRelocs(); }

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
    TORCH_CHECK(inputs.size() == loads_.size(), "GraphSplitKernel expects ", loads_.size(), " inputs, got ",
                inputs.size());
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
    size_t sym_scalar_index = 0;
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
            sym_scalar_input_[sym_scalar_index++]->data_ =
                static_cast<float>(inputs[i].cast<c10::SymFloat>().expect_float());
        } else if (py::isinstance<c10::SymInt>(inputs[i])) {
            sym_scalar_input_[sym_scalar_index++]->data_ = inputs[i].cast<c10::SymInt>().expect_int();
        } else if (py::isinstance<py::float_>(inputs[i])) {
            sym_scalar_input_[sym_scalar_index++]->data_ = inputs[i].cast<float>();
        } else if (py::isinstance<py::int_>(inputs[i])) {
            sym_scalar_input_[sym_scalar_index++]->data_ = inputs[i].cast<int64_t>();
        } else {
            TORCH_CHECK(false, "Unsupported dynamic input type for DynGraphSplitKernel.");
        }
    }
    UpdateSymShapeData();
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

int TorchKernelPy::Launch(void** addr, aclrtStream stream)
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
            auto sym_ptr = const_input_.emplace_back(std::make_shared<ScalarRefPy>());
            sym_ptr->data_ = shape_array[i].cast<int64_t>();
            sym_shape.emplace_back(sym_ptr);
        } else {
            sym_shape.emplace_back(shape_array[i].cast<ScalarRefPyPtr>());
        }
    }
    return ref;
}

void DynKernelPy::UpdateSymShapeData()
{
    for (size_t i = 0; i < sym_shape_.size(); i++) {
        for (size_t j = 0; j < shapes_[i]->size; j++) {
            shapes_[i]->shape_data[j] = sym_shape_[i][j]->data_.i64;
        }
    }
}

void DynKernelPy::Setup() { SetupRelocs(); }

py::object DynKernelPy::Call(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    const auto num_sym = sym_scalar_input_.size();
    TORCH_CHECK(args.size() == num_inputs + num_outputs + num_sym, "DynKernel expects ",
                num_inputs + num_outputs + num_sym, " args, got ", args.size());
    std::vector<at::Tensor> tensor_list;
    std::vector<std::pair<at::Tensor, at::Tensor> > out_refs;

    auto info = std::make_shared<DynamicInfo>();
    info->shape.reserve(num_inputs);
    info->strides.reserve(num_inputs);
    info->scalars.reserve(num_sym);
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
                    TORCH_CHECK(!tensor.is_contiguous(), "Expected non-contiguous output tensor at index ", i);
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
            info->scalars.emplace_back(static_cast<float>(args[i].cast<c10::SymFloat>().expect_float()));
        } else if (py::isinstance<c10::SymInt>(args[i])) {
            info->scalars.emplace_back(args[i].cast<c10::SymInt>().expect_int());
        } else if (py::isinstance<py::float_>(args[i])) {
            info->scalars.emplace_back(args[i].cast<float>());
        } else if (py::isinstance<py::int_>(args[i])) {
            info->scalars.emplace_back(args[i].cast<int64_t>());
        } else {
            TORCH_CHECK(false, "Unsupported dynamic input type for DynKernel.");
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
        for (size_t i = 0; i < sym_scalar_input_.size(); i++) {
            TORCH_CHECK(sym_scalar_input_[i]->data_.type == info->scalars[i].type, "Scalar type mismatch at index ", i,
                        ": expected ", sym_scalar_input_[i]->data_.type, ", got ", info->scalars[i].type);
            if (sym_scalar_input_[i]->data_.type == kInt64) {
                sym_scalar_input_[i]->data_ = info->scalars[i].i64;
            } else {
                sym_scalar_input_[i]->data_ = info->scalars[i].f32;
            }
        }
        UpdateSymShapeData();
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
