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

namespace dvm {
class NDObjectPy {
public:
    explicit NDObjectPy(NDObject* obj) : obj_(obj) {}
    NDObject* Get() const { return obj_; }

private:
    NDObject* obj_;
};

struct NDSymInt {
    explicit NDSymInt(int32_t data) { data_ = data; }
    ScalarRef data_;
};

struct NDSymFloat {
    explicit NDSymFloat(float data) { data_ = data; }
    ScalarRef data_;
};

using NDOpPyPtr = std::shared_ptr<NDObjectPy>;
using NDSymIntPtr = std::shared_ptr<NDSymInt>;
using NDSymFloatPtr = std::shared_ptr<NDSymFloat>;

class KernelPy {
public:
    KernelPy(const KernelType& kernel_type, uint32_t flags);
    ~KernelPy();

    virtual py::object Load(at::IntArrayRef shape, at::ScalarType dtype);
    virtual py::object ViewLoad(at::IntArrayRef shape, at::IntArrayRef stride, at::ScalarType dtype);
    py::object Store(py::object obj, py::object dtype);
    void SetStoreInplace(py::object store);
    template <UnaryOpType op_type> py::object Unary(py::object input);
    template <BinaryOpType op_type> py::object Binary(py::object lhs, py::object rhs);
    py::object Broadcast(py::object input, py::object shape);
    py::object BroadcastScalar(py::object scalar, py::object shape, at::ScalarType dtype);
    py::object Reshape(py::object input, py::object shape);
    py::object Cast(py::object input, at::ScalarType dtype);
    template <ReduceOpType op_type> py::object Reduce(py::object input, py::object dims, bool keepdims);
    py::object Select(py::object cond, py::object lhs, py::object rhs);
    py::object ElementAny(py::object input);
    py::object Copy(py::object input);
    py::object OneHot(py::object indices, int depth, int axis, c10::Scalar on_value, c10::Scalar off_value,
                      at::ScalarType dtype);
    py::object MatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias);
    py::object GroupedMatMul(py::object lhs, py::object rhs, bool trans_a, bool trans_b, py::object bias,
                             py::object group_list, int64_t group_type, int64_t group_list_type);
    void ParallelNext();
    void SpecNext();
    at::ScalarType GetDtype(py::object op);

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
    std::string DisAssemble();
    std::string DumpGraph();

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
    Kernel kernel_;
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
class DynKernelPy : public KernelPy {
public:
    DynKernelPy(const KernelType& kernel_type, uint32_t flags) : KernelPy(kernel_type, flags) {}
    ~DynKernelPy();
    py::object Load(at::IntArrayRef shape, at::ScalarType dtype) override;
    py::object ViewLoad(at::IntArrayRef shape, at::IntArrayRef stride, at::ScalarType dtype) override;
    struct LoadShapeRef {
        ShapeRef shape;
        ShapeRef stride;
    };
    LoadShapeRef* GetDynLoadShapeRef(size_t dim_size);
    void Setup() override;
    py::object Call(py::args args) override;
    ShapeRef* SymIntArraytoShapeRef(py::object shape) override;

    py::object MakeNDSymInt() { return py::cast(sym_input_.emplace_back(std::make_shared<NDSymInt>(-1))); }
    py::object MakeNDSymFloat() { return py::cast(sym_float_input_.emplace_back(std::make_shared<NDSymFloat>(0.0f))); }

    struct DynamicInfo {
        std::vector<void*> addr;
        std::vector<int32_t> symint;
        std::vector<float> symfloat;
        std::vector<std::vector<int64_t> > shape;
        std::vector<std::vector<int64_t> > strides;
    };

protected:
    std::vector<LoadShapeRef*> dyn_load_shapes_;
    std::vector<NDSymIntPtr> sym_input_;
    std::vector<NDSymFloatPtr> sym_float_input_;
    std::vector<NDSymIntPtr> const_input_;
    std::vector<std::vector<NDSymIntPtr> > sym_shape_;
};

class GraphSplitKernelPy : public KernelPy, public GraphSplitBase {
public:
    GraphSplitKernelPy()
        : KernelPy(KernelType::kSplit, static_cast<uint32_t>(KernelFlag::kUnifyWS))
    {
    }
    void Setup() override;
    py::object Call(py::args inputs) override;
};

class DynGraphSplitKernelPy : public DynKernelPy, public GraphSplitBase {
public:
    DynGraphSplitKernelPy()
        : DynKernelPy(
              KernelType::kSplit,
              static_cast<uint32_t>(KernelFlag::kUnifyWS | KernelFlag::kDynamic))
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
