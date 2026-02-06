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
#include <cstdint>
#include <memory>
#include <vector>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "third_party/dvm/dvm/include/dvm.h"
#include "third_party/dvm/dvm/include/dvm_py.h"


namespace dvm {
class TorchKernelPy : public KernelPy {
public:
    TorchKernelPy(const KernelType& kernel_type, uint32_t flags);
    ~TorchKernelPy();

    py::object Load(py::object shape, DataTypePy type) override;
    py::object ViewLoad(py::object shape, py::object stride, int64_t offset, DataTypePy type) override;
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

    int Launch(void** addr, aclrtStream stream);
    virtual void Setup();
    virtual py::object Call(py::args args);

    static void SetDeterm(bool enable);
    static void SetTuning(bool enable);
    virtual ShapeRef* SymIntArraytoShapeRef(py::object shape);
    ShapeRef* SymIntArraytoShapeRef(at::IntArrayRef shape_array);
    py::object CreateOutputs(const at::TensorOptions& options, void** addr);
    void SetupRelocs();

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
    std::vector<RelocEntry> relocs_;

    std::vector<ShapeWithRef*> shapes_;
    std::vector<NDObject*> loads_;
    std::vector<NDObject*> stores_;
    std::vector<bool> contiguity_flags_;
    size_t ws_size_;
    std::string op_name_;
    std::string op_fullname_;
};

class GraphSplitBase : public WsAllocator {
public:
    virtual ~GraphSplitBase() = default;
    int Launch(Kernel& kernel, void** addr, aclrtStream stream, std::vector<RelocEntry>& relocs);

    void* Alloc(size_t size) override { return AllocWorkspace(size); }

    inline void* AllocWorkspace(uint64_t size)
    {
        ws_ = at_npu::native::allocate_workspace(size, stream_);
        return const_cast<void*>(ws_.storage().data());
    }

protected:
    at::Tensor ws_;
    aclrtStream stream_;
};
class DynKernelPy : public TorchKernelPy {
public:
    DynKernelPy(const KernelType& kernel_type, uint32_t flags) : TorchKernelPy(kernel_type, flags) {}
    ~DynKernelPy();
    py::object Load(py::object shape, DataTypePy type) override;
    py::object ViewLoad(py::object shape, py::object stride, int64_t offset, DataTypePy type) override;
    struct LoadShapeRef {
        ShapeRef shape;
        ShapeRef stride;
    };
    LoadShapeRef* GetDynLoadShapeRef(size_t dim_size);
    void Setup() override;
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
    std::vector<LoadShapeRef*> dyn_load_shapes_;
    std::vector<ScalarRefPyPtr> sym_scalar_input_;
    std::vector<ScalarRefPyPtr> const_input_;
    std::vector<std::vector<ScalarRefPyPtr> > sym_shape_;
};

class GraphSplitKernelPy : public TorchKernelPy, public GraphSplitBase {
public:
    GraphSplitKernelPy() : TorchKernelPy(KernelType::kSplit, static_cast<uint32_t>(KernelFlag::kUnifyWS)) {}
    void Setup() override;
    py::object Call(py::args inputs) override;
};

class DynGraphSplitKernelPy : public DynKernelPy, public GraphSplitBase {
public:
    DynGraphSplitKernelPy()
        : DynKernelPy(KernelType::kSplit, static_cast<uint32_t>(KernelFlag::kUnifyWS | KernelFlag::kDynamic))
    {
    }
    void Setup() override;
    py::object Call(py::args inputs) override;
    volatile int64_t entry_cnt_ = 0;
    volatile int64_t exit_cnt_ = 0;
};
} // namespace dvm
#endif // BUILD_LIBTORCH
#endif // _DVM_PYBIND_API_H_
