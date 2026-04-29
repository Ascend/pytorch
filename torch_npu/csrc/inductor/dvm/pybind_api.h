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

#ifndef _DVM_PYBIND_API_H_
#define _DVM_PYBIND_API_H_
#ifndef BUILD_LIBTORCH
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <unordered_map>
#include <vector>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "third_party/dvm/dvm/include/dvm.h"
#include "third_party/dvm/dvm/include/dvm_py.h"


namespace dvm {
struct SplitCloneHelper : public dvm::CloneHelper {
    IntArrayRef* GetClone(IntArrayRef* shape) override
    {
        auto it = ref_map_.find(shape);
        return it != ref_map_.end() ? static_cast<IntArrayRef*>(it->second) : shape;
    }

    ScalarRef* GetClone(ScalarRef* scalar) override
    {
        auto it = ref_map_.find(scalar);
        return it != ref_map_.end() ? static_cast<ScalarRef*>(it->second) : scalar;
    }

    NDObject* GetClone(NDObject* op) override
    {
        auto it = op_map_.find(op);
        return it != op_map_.end() ? it->second : nullptr;
    }

    void SetClone(NDObject* op, NDObject* clone) override { op_map_[op] = clone; }

    std::unordered_map<NDObject*, NDObject*> op_map_;
    std::unordered_map<void*, void*> ref_map_;
};
class TorchKernelPy : public KernelPy {
public:
    TorchKernelPy(int kernel_type, uint32_t flags);
    ~TorchKernelPy();

    py::object Load(py::object shape, DataTypePy type) override;
    py::object ViewLoad(py::object shape, py::object stride, DataTypePy type) override;
    py::object Store(py::object obj, DataTypePy type) override;
    IntArrayRef* GetShapeRef(py::object shape) override;

    void SetKernelInfo(const std::string& op_name, const std::string& op_fullname,
                       const std::vector<bool>& contiguity_flags)
    {
        contiguity_flags_ = contiguity_flags;
        op_name_ = op_name;
        op_fullname_ = op_fullname;
        kernel_.SetNameHint(op_name_.c_str(), op_fullname_.c_str());
    }

    int LaunchV1(void** addr, aclrtStream stream, void* workspace_ptr);
    int LaunchV2(void** addr, aclrtStream stream);
    virtual void Setup();
    virtual py::object Run(py::args args);
    virtual py::object Call(py::args args);

    static void SetDeterm(bool enable);
    static void SetTuning(bool enable);
    virtual ShapeRef* SymIntArraytoShapeRef(py::object shape);
    ShapeRef* SymIntArraytoShapeRef(at::IntArrayRef shape_array);
    py::object CreateOutputs(const at::TensorOptions& options, void** addr);
    void SetupRelocs();
    void SetWorkspaceSize(size_t size) { ws_size_ = size; }

    struct ShapeWithRef : public ShapeRef {
        enum { MAX_SIZE = 8 };
        ShapeWithRef(size_t sz)
        {
            data = shape_data;
            size = sz;
        }
        int64_t shape_data[MAX_SIZE];
    };

protected:
    struct ParsedCallInputs {
        std::shared_ptr<std::vector<void*> > addr;
        at::TensorOptions options;
    };

    ParsedCallInputs ParseTensorCallInputs(py::args inputs, std::vector<at::Tensor>& tensor_refs) const;

    std::vector<RelocEntry> relocs_;

    std::vector<ShapeWithRef*> shapes_;
    std::vector<NDObject*> loads_;
    std::vector<NDObject*> stores_;
    std::vector<at::Tensor> tensor_list_;
    std::vector<std::pair<at::Tensor, at::Tensor> > out_refs_;
    std::vector<bool> contiguity_flags_;
    size_t ws_size_;
    int kernel_type_;
    uint32_t kernel_flags_;
    std::string op_name_;
    std::string op_fullname_;
};

class GraphSplitBase : public WsAllocator {
public:
    virtual ~GraphSplitBase() = default;
    int LaunchV1(Kernel& kernel, void** addr, aclrtStream stream, std::vector<RelocEntry>& relocs,
                 void* workspace_ptr);
    int LaunchV2(Kernel& kernel, void** addr, aclrtStream stream, std::vector<RelocEntry>& relocs);

    void* Alloc(size_t size) override;

protected:
    at::Tensor ws_;
    aclrtStream stream_;
};
class DynKernelPy : public TorchKernelPy {
public:
    DynKernelPy(int kernel_type, uint32_t flags) : TorchKernelPy(kernel_type, flags) {}
    ~DynKernelPy();
    py::object Load(py::object shape, DataTypePy type) override;
    py::object ViewLoad(py::object shape, py::object stride, DataTypePy type) override;
    struct LoadShapeRef {
        ShapeRef shape;
        ShapeRef stride;
    };
    LoadShapeRef* GetDynLoadShapeRef(size_t dim_size);
    void Setup() override;
    py::object Run(py::args args) override;
    py::object Call(py::args args) override;
    ShapeRef* SymIntArraytoShapeRef(py::object shape) override;
    void UpdateSymShapeData();

    py::object MakeScalar(DataTypePy type = DataTypePy(kDataTypeEnd))
    {
        auto scalar = std::make_shared<ScalarRefPy>(type);
        sym_scalar_input_.emplace_back(scalar);
        return py::cast(scalar);
    }

    struct DynamicInfo {
        std::vector<void*> addr;
        std::vector<ScalarRef> scalars;
        std::vector<std::vector<int64_t> > shape;
        std::vector<std::vector<int64_t> > strides;
    };

protected:
    ParsedCallInputs ParseDynCallInputs(py::args inputs, std::vector<at::Tensor>& tensor_refs) const;
    DynKernelPy* AcquireExecutor()
    {
        std::lock_guard<std::mutex> lock(dyn_executor_mutex_);
        if (!dyn_executors_.empty()) {
            auto* executor = dyn_executors_.back();
            dyn_executors_.pop_back();
            return executor;
        }
        auto new_executor = CloneExecutor();
        auto* executor = new_executor.get();
        dyn_owned_executors_.push_back(std::move(new_executor));
        return executor;
    }
    void ReleaseExecutor(DynKernelPy* executor)
    {
        std::lock_guard<std::mutex> lock(dyn_executor_mutex_);
        dyn_executors_.push_back(executor);
    }
    virtual std::unique_ptr<DynKernelPy> CloneExecutor() const;
    void CloneExecutorStateTo(DynKernelPy& executor) const;

    std::vector<LoadShapeRef*> dyn_load_shapes_;
    std::vector<ScalarRefPyPtr> sym_scalar_input_;
    std::vector<ScalarRefPyPtr> const_input_;
    std::vector<std::vector<ScalarRefPyPtr> > sym_shape_;

private:
    std::mutex dyn_executor_mutex_;
    std::vector<DynKernelPy*> dyn_executors_;
    std::vector<std::unique_ptr<DynKernelPy>> dyn_owned_executors_;
};

class GraphSplitKernelPy : public TorchKernelPy, public GraphSplitBase {
public:
    GraphSplitKernelPy() : TorchKernelPy(KernelPy::K_SPLIT, KernelPy::F_UWS) {}
    void Setup() override;
    py::object Run(py::args inputs) override;
    py::object Call(py::args inputs) override;
};

class DynGraphSplitKernelPy : public DynKernelPy, public GraphSplitBase {
public:
    DynGraphSplitKernelPy()
        : DynKernelPy(KernelPy::K_SPLIT, KernelPy::F_UWS | KernelPy::F_DYN)
    {
    }
    void Setup() override;
    py::object Run(py::args inputs) override;
    py::object Call(py::args inputs) override;

private:
    std::unique_ptr<DynKernelPy> CloneExecutor() const override;
};
} // namespace dvm
#endif // BUILD_LIBTORCH
#endif // _DVM_PYBIND_API_H_
