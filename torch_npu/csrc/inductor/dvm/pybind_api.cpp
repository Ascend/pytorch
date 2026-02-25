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

    pybind11::class_<TorchKernelPy, KernelPy, std::shared_ptr<TorchKernelPy> >(dvm_m, "TorchKernel")
        .def(py::init<int, uint32_t>())
        .def("set_kernel_info", &TorchKernelPy::SetKernelInfo, "set_kernel_info")
        .def("setup", &TorchKernelPy::Setup, "setup")
        .def("run", &TorchKernelPy::Run, "run kernel")
        .def("__call__", &TorchKernelPy::Call, "call kernel");

    pybind11::class_<DynKernelPy, TorchKernelPy, std::shared_ptr<DynKernelPy> >(dvm_m, "DynKernel")
        .def(py::init<int, uint32_t>())
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

TorchKernelPy::TorchKernelPy(int kernel_type, uint32_t flags)
    : ws_size_(0), kernel_type_(kernel_type), kernel_flags_(flags)
{
    kernel_.Reset(static_cast<KernelType>(kernel_type), flags);
}

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

py::object TorchKernelPy::ViewLoad(py::object shape, py::object stride, DataTypePy type)
{
    ShapeRef* shape_ref = SymIntArraytoShapeRef(shape);
    ShapeRef* stride_ref = SymIntArraytoShapeRef(stride);
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, type);
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

py::object DynKernelPy::ViewLoad(py::object shape, py::object stride, DataTypePy type)
{
    auto shape_seq = shape.cast<py::sequence>();
    auto ref = GetDynLoadShapeRef(shape_seq.size());
    ref->stride = ref->shape;
    ShapeRef* shape_ref = &ref->shape;
    ShapeRef* stride_ref = &ref->stride;
    auto op = kernel_.Load(nullptr, shape_ref, stride_ref, type);
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

TorchKernelPy::ParsedCallInputs TorchKernelPy::ParseTensorCallInputs(py::args inputs,
                                                                     std::vector<at::Tensor>& tensor_refs) const
{
    TORCH_CHECK(inputs.size() == loads_.size(), "Call expects ", loads_.size(), " input tensors, got ", inputs.size());

    auto addr = std::make_shared<std::vector<void*> >();
    addr->resize(relocs_.size());
    tensor_refs.reserve(loads_.size());
    at::TensorOptions options;

    for (size_t i = 0; i < loads_.size(); ++i) {
        auto tensor = inputs[i].cast<at::Tensor>();
        (*addr)[i] = tensor.data_ptr();
        options = tensor.options();
        tensor_refs.emplace_back(tensor);
    }
    return {addr, options};
}

py::object TorchKernelPy::Run(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    TORCH_CHECK(args.size() == num_inputs + num_outputs, "DVM kernel run expects ", num_inputs + num_outputs,
                " tensors, got ", args.size());
    std::vector<at::Tensor> tensor_list;
    std::vector<std::pair<at::Tensor, at::Tensor> > out_refs;
    auto addr = std::make_shared<std::vector<void*> >();
    tensor_list.reserve(args.size());
    out_refs.reserve(num_outputs);
    addr->resize(num_inputs + num_outputs);
    for (size_t i = 0; i < args.size(); i++) {
        auto tensor = args[i].cast<at::Tensor>();
        if (!contiguity_flags_.empty() && !contiguity_flags_[i]) {
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

py::object TorchKernelPy::Call(py::args args)
{
    const auto num_inputs = loads_.size();
    const auto num_outputs = stores_.size();
    std::vector<at::Tensor> tensor_list;
    auto parsed = ParseTensorCallInputs(args, tensor_list);
    auto ret = CreateOutputs(parsed.options, parsed.addr->data() + num_inputs);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto dvm_call = [this, addr = parsed.addr, stream]() { return Launch(addr->data(), stream); };
    at_npu::native::OpCommand::RunOpApiV2(op_name_, dvm_call);
    return ret;
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

std::unique_ptr<DynKernelPy> DynGraphSplitKernelPy::CloneExecutor() const
{
    auto executor = std::make_unique<DynGraphSplitKernelPy>();
    CloneExecutorStateTo(*executor);
    return executor;
}

void DynGraphSplitKernelPy::Setup() { DynKernelPy::Setup(); }

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

py::object GraphSplitKernelPy::Run(py::args)
{
    TORCH_CHECK(false, "GraphSplitKernel::Run is unsupported. Use Call/__call__ to create outputs internally.");
}

py::object GraphSplitKernelPy::Call(py::args inputs)
{
    std::vector<at::Tensor> tensor_list;
    auto parsed = ParseTensorCallInputs(inputs, tensor_list);
    auto ret = CreateOutputs(parsed.options, parsed.addr->data() + loads_.size());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch_call = [this, stream, addr = parsed.addr]() -> int {
        return GraphSplitBase::Launch(kernel_, addr->data(), stream, relocs_);
    };
    at_npu::native::OpCommand::RunOpApiV2(op_name_, launch_call);
    return ret;
}

py::object DynGraphSplitKernelPy::Run(py::args)
{
    TORCH_CHECK(false, "DynGraphSplitKernel::Run is unsupported. Use Call/__call__ to create outputs internally.");
}

py::object DynGraphSplitKernelPy::Call(py::args inputs)
{
    auto* executor = static_cast<DynGraphSplitKernelPy*>(AcquireExecutor());
    std::vector<at::Tensor> tensor_list;
    auto parsed = executor->ParseDynCallInputs(inputs, tensor_list);

    executor->UpdateSymShapeData();
    executor->kernel_.Infer();
    auto ret = executor->CreateOutputs(parsed.options, parsed.addr->data() + executor->loads_.size());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch_call = [this, executor, stream, addr = parsed.addr]() -> int {
        int ret = executor->GraphSplitBase::Launch(executor->kernel_, addr->data(), stream, executor->relocs_);
        ReleaseExecutor(executor);
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

std::unique_ptr<DynKernelPy> DynKernelPy::CloneExecutor() const
{
    auto executor = std::make_unique<DynKernelPy>(kernel_type_, kernel_flags_);
    CloneExecutorStateTo(*executor);
    return executor;
}

void DynKernelPy::CloneExecutorStateTo(DynKernelPy& executor) const
{
    SplitCloneHelper helper;
    std::unordered_map<ScalarRef*, ScalarRefPyPtr> scalar_to_owner;

    executor.ws_size_ = ws_size_;
    executor.op_name_ = op_name_;
    executor.op_fullname_ = op_fullname_;
    executor.contiguity_flags_ = contiguity_flags_;
    executor.kernel_.SetNameHint(executor.op_name_.c_str(), executor.op_fullname_.c_str());

    executor.shapes_.reserve(shapes_.size());
    for (auto ref : shapes_) {
        auto clone_ref = new ShapeWithRef(ref->size);
        for (size_t i = 0; i < ref->size; ++i) {
            clone_ref->shape_data[i] = ref->shape_data[i];
        }
        executor.shapes_.push_back(clone_ref);
        helper.ref_map_[ref] = clone_ref;
    }

    executor.dyn_load_shapes_.reserve(dyn_load_shapes_.size());
    for (auto ref : dyn_load_shapes_) {
        auto clone_ref = new LoadShapeRef();
        clone_ref->shape = ref->shape;
        clone_ref->stride = ref->stride;
        executor.dyn_load_shapes_.push_back(clone_ref);
        helper.ref_map_[&ref->shape] = &clone_ref->shape;
        helper.ref_map_[&ref->stride] = &clone_ref->stride;
    }

    auto clone_scalar = [&helper, &scalar_to_owner](const ScalarRefPyPtr& src, std::vector<ScalarRefPyPtr>& dst) {
        auto clone = std::make_shared<ScalarRefPy>();
        clone->data_ = src->data_;
        helper.ref_map_[&src->data_] = &clone->data_;
        scalar_to_owner[&clone->data_] = clone;
        dst.push_back(clone);
    };
    for (const auto& scalar : const_input_) {
        clone_scalar(scalar, executor.const_input_);
    }
    for (const auto& scalar : sym_scalar_input_) {
        clone_scalar(scalar, executor.sym_scalar_input_);
    }

    executor.kernel_.Clone(kernel_, helper);
    executor.loads_.reserve(loads_.size());
    for (auto op : loads_) {
        executor.loads_.push_back(helper.GetClone(op));
    }
    executor.stores_.reserve(stores_.size());
    for (auto op : stores_) {
        executor.stores_.push_back(helper.GetClone(op));
    }

    executor.sym_shape_.reserve(sym_shape_.size());
    for (const auto& shape_scalars : sym_shape_) {
        auto& clone_shape_scalars = executor.sym_shape_.emplace_back();
        clone_shape_scalars.reserve(shape_scalars.size());
        for (const auto& scalar : shape_scalars) {
            auto it = helper.ref_map_.find(&scalar->data_);
            TORCH_CHECK(it != helper.ref_map_.end(), "Failed to clone symbolic shape scalar reference.");
            auto clone_scalar_ref = static_cast<ScalarRef*>(it->second);
            auto owner_it = scalar_to_owner.find(clone_scalar_ref);
            TORCH_CHECK(owner_it != scalar_to_owner.end(), "Failed to remap cloned symbolic shape scalar.");
            clone_shape_scalars.push_back(owner_it->second);
        }
    }

    executor.SetupRelocs();
}

TorchKernelPy::ParsedCallInputs DynKernelPy::ParseDynCallInputs(py::args inputs,
                                                                std::vector<at::Tensor>& tensor_refs) const
{
    TORCH_CHECK(inputs.size() == dyn_load_shapes_.size() + sym_scalar_input_.size(), "Dynamic call expects ",
                dyn_load_shapes_.size() + sym_scalar_input_.size(), " args, got ", inputs.size());

    auto addr = std::make_shared<std::vector<void*> >();
    addr->resize(relocs_.size());
    tensor_refs.reserve(dyn_load_shapes_.size());
    at::TensorOptions options;

    size_t input_index = 0;
    size_t sym_scalar_index = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (THPVariable_Check(inputs[i].ptr())) {
            auto tensor = inputs[i].cast<at::Tensor>();
            (*addr)[input_index] = tensor.data_ptr();
            auto ref = dyn_load_shapes_[input_index++];
            if (ref->stride.data) {
                ref->stride.data = tensor.strides().data();
            }
            ref->shape.data = tensor.sizes().data();
            options = tensor.options();
            tensor_refs.emplace_back(tensor);
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
            const char* py_type = inputs[i].ptr() ? Py_TYPE(inputs[i].ptr())->tp_name : "<null>";
            TORCH_CHECK(
                false, "Unsupported dynamic input type at arg[", i,
                "]. Expected one of: Tensor, c10::SymFloat, c10::SymInt, float, int. Got Python type: ", py_type);
        }
    }

    TORCH_CHECK(input_index == dyn_load_shapes_.size(), "Dynamic call expects ", dyn_load_shapes_.size(),
                " tensor inputs, got ", input_index);
    TORCH_CHECK(sym_scalar_index == sym_scalar_input_.size(), "Dynamic call expects ", sym_scalar_input_.size(),
                " scalar inputs, got ", sym_scalar_index);
    return {addr, options};
}

void DynKernelPy::Setup()
{
    SetupRelocs();
    dyn_executors_.push_back(this);
}

py::object DynKernelPy::Run(py::args args)
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
            if (!contiguity_flags_.empty() && !contiguity_flags_[i]) {
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
            if (info->scalars[i].type == kInt64) {
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

py::object DynKernelPy::Call(py::args args)
{
    auto* executor = AcquireExecutor();
    const auto num_inputs = executor->loads_.size();
    std::vector<at::Tensor> tensor_list;
    auto parsed = executor->ParseDynCallInputs(args, tensor_list);

    executor->UpdateSymShapeData();
    executor->ws_size_ = executor->kernel_.CodeGen();
    auto ret = executor->CreateOutputs(parsed.options, parsed.addr->data() + num_inputs);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto dvm_call = [this, executor, addr = parsed.addr, stream]() {
        int ret = executor->Launch(addr->data(), stream);
        ReleaseExecutor(executor);
        return ret;
    };
    at_npu::native::OpCommand::RunOpApiV2(executor->op_name_, dvm_call);
    return ret;
}
} // namespace dvm
#endif // BUILD_LIBTORCH
