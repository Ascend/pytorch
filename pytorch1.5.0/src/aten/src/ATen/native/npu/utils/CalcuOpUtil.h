// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NATIVE_NPU_UTILS_CALCU_OP_UTIL__
#define __NATIVE_NPU_UTILS_CALCU_OP_UTIL__

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/npu/utils/NpuUtils.h>
#include <ATen/npu/Exceptions.h>
#include <c10/npu/npu_log.h>
#include <ATen/native/npu/frame/NPUDefine.h>
#include <c10/npu/interface/AclInterface.h>
#include <stdint.h>
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define ASCEND_LIKELY(expr)    (__builtin_expect(static_cast<bool>(expr), 1))
#define ASCEND_UNLIKELY(expr)  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define ASCEND_LIKELY(expr)    (expr)
#define ASCEND_UNLIKELY(expr)  (expr)
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define ASCEND_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#elif defined(_MSC_VER)
#define ASCEND_ALWAYS_INLINE __forceinline
#else
#define ASCEND_ALWAYS_INLINE inline
#endif

#define ACL_REQUIRE_OK_OP(expr, opstr)            \
  do {                                            \
    if (ASCEND_UNLIKELY((expr) != 0)) {           \
      printf("%s\n", opstr);                      \
      TORCH_CHECK(                                \
          (expr) == 0,                            \
          __func__,                               \
          ":",                                    \
          __FILE__,                               \
          ":",                                    \
          __LINE__,                               \
          " NPU error,NPU error code is:",        \
          expr, "\n",                             \
          c10::npu::acl::AclGetErrMsg());         \
    }                                             \
  } while (0)

namespace at {
namespace native {
namespace npu {

using StorageAndOffsetPair = std::pair<const StorageImpl*, int64_t>;

class NPUTensorDesc {
 public:
  enum TensorDescType {
    TENSOR = 0,
    SCALAR = 1,
    TENSOR_SCALAR = 2, // dim = 0 tensor
    NONE_TENSOR = 3, // None Tensor
  };

 public:
  NPUTensorDesc() {}
  ~NPUTensorDesc() = default;

  explicit NPUTensorDesc(const Tensor& tensor)
      : tensor(tensor), tensorDescType(TensorDescType::TENSOR) {}

  explicit NPUTensorDesc(const Scalar& scalar)
      : scalar(scalar), tensorDescType(TensorDescType::SCALAR) {}

  NPUTensorDesc(const Scalar& scalar, ScalarType scalarDataType)
      : scalar(scalar),
        scalarType(scalarDataType),
        tensorDescType(TensorDescType::SCALAR) {}

 public:
  Tensor tensor;
  Scalar scalar;
  ScalarType scalarType = ScalarType::Undefined;
  TensorDescType tensorDescType;
  string tensorDescName;
  string realDataType;
};

class NPUAttrDesc {
 public:
  enum AttrDescType {
    BOOL_TYPE = 0,
    INT_TYPE,
    FLOAT_TYPE,
    STRING_TYPE,
    LIST_BOOL_TYPE,
    LIST_INT_TYPE,
    LIST_FLOAT_TYPE,
    LIST_STRING_TYPE,
    LIST_LIST_INT_TYPE,
  };

  NPUAttrDesc(string attrName, bool attrValue)
      : attrName(attrName), boolAttrValue(attrValue) {
    attrType = AttrDescType::BOOL_TYPE;
  }

  NPUAttrDesc(string attrName, int64_t attrValue)
      : attrName(attrName), intAttrValue(attrValue) {
    attrType = AttrDescType::INT_TYPE;
  }

  NPUAttrDesc(string attrName, float attrValue)
      : attrName(attrName), floatAttrValue(attrValue) {
    attrType = AttrDescType::FLOAT_TYPE;
  }

  NPUAttrDesc(string attrName, string attrValue)
      : attrName(attrName), stringAttrValue(attrValue) {
    attrType = AttrDescType::STRING_TYPE;
  }

  NPUAttrDesc(string attrName, IntArrayRef attrValue) : attrName(attrName) {
    for (int i = 0; i < attrValue.size(); i++) {
      listIntAttrValue.emplace_back(attrValue[i]);
    }
    attrType = AttrDescType::LIST_INT_TYPE;
  }

  NPUAttrDesc(string attrName, at::ArrayRef<float> attrValue)
      : attrName(attrName) {
    for (int i = 0; i < attrValue.size(); i++) {
      listFloatAttrValue.emplace_back(attrValue[i]);
    }
    attrType = AttrDescType::LIST_FLOAT_TYPE;
  }

  NPUAttrDesc(string attrName, at::ArrayRef<IntArrayRef> attrValue)
      : attrName(attrName) {
    SmallVector<int64_t, N> listInt;

    for (int i = 0; i < attrValue.size(); i++) {
      for (int j = 0; j < attrValue[i].size(); j++) {
        listInt.emplace_back(attrValue[i][j]);
      }
      listListIntAttrListIntVal.emplace_back(listInt);
      listInt.clear();

      listListIntAttrValue.emplace_back(listListIntAttrListIntVal[i].data());
      listListIntAttrListIntNum.emplace_back(attrValue[i].size());
    }
    attrType = AttrDescType::LIST_LIST_INT_TYPE;
  }

  ~NPUAttrDesc() = default;

 public:
  string attrName;
  AttrDescType attrType;
  bool boolAttrValue = false;
  int64_t intAttrValue = 0;
  float floatAttrValue = 0.0;
  string stringAttrValue;
  SmallVector<int64_t, N> listIntAttrValue;
  SmallVector<float, N> listFloatAttrValue;
  SmallVector<int64_t*, N>
      listListIntAttrValue; // Pointer to values of each listInt.
  SmallVector<int, N>
      listListIntAttrListIntNum; // Pointer to number of each listInt.
  SmallVector<SmallVector<int64_t, N>, N>
      listListIntAttrListIntVal; // Value of each listInt.
};

class CalcuOpUtil {
 public:
  static aclDataType convert_to_acl_data_type(const ScalarType data_type);
  static aclDataType convert_to_acl_data_type(
      const ScalarType data_type,
      const string& realDataType);
  static Scalar ConvertTensorToScalar(const Tensor& tensor);
  static Tensor CopyScalarToDevice(
      const Scalar& cpu_scalar,
      ScalarType scalar_data_type);
  static Tensor copy_tensor_host_to_device(const Tensor& cpu_tensor);
  static NPUStatus AclrtMemcpyAsync(
      const std::pair<Tensor, int64_t>& dst,
      size_t dst_size,
      const std::pair<Tensor, int64_t>& src,
      size_t src_size,
      aclrtMemcpyKind kind);

  // Add some public interfaces for aclrtmemcpy process,
  // to launch graph in graph mode automatically.
  TORCH_NPU_API static aclError AclrtMemcpyAsyncWithModeSwitch(
      const StorageAndOffsetPair& dst,
      size_t dstMax,
      const StorageAndOffsetPair& src,
      size_t count,
      aclrtMemcpyKind kind,
      aclrtStream stream);
  TORCH_NPU_API static aclError AclrtMemcpyAsyncWithModeSwitch(
      const StorageAndOffsetPair& dst,
      size_t dstMax,
      const void* src,
      size_t count,
      aclrtMemcpyKind kind,
      aclrtStream stream);
  TORCH_NPU_API static aclError AclrtMemcpyAsyncWithModeSwitch(
      void* dst,
      size_t dstMax,
      const StorageAndOffsetPair& src,
      size_t count,
      aclrtMemcpyKind kind,
      aclrtStream stream);
  TORCH_NPU_API static aclError LaunchAsyncCopyTaskWithModeSwitch(
      const Tensor& dst,
      size_t dstMax,
      const Tensor& src,
      size_t count,
      aclrtMemcpyKind kind);
  TORCH_NPU_API static aclError LaunchAsyncCopyTaskWithModeSwitch(
      const StorageImpl& dst,
      size_t dstMax,
      void* src,
      size_t count,
      aclrtMemcpyKind kind);

  static void check_memory_over_laps(
      SmallVector<Tensor, N>& inputs,
      SmallVector<Tensor, N>& outputs);
  static int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr);
  static bool is_transpose_last_two_dims(const Tensor& tensor);
  static bool is_scalar_wrapped_to_tensor(const Tensor& tensor);
  static bool is_scalar_one(const Scalar& scalar);
  static float get_scalar_float_value(const Scalar& scalar);
  static int64_t get_tensor_npu_format(const Tensor& tensor);
  static ScalarType GetNPUTensorDescScalarType(
      const NPUTensorDesc& npuTensorDesc);
  static SmallVector<Tensor, N> ConvertTensorListToSmallVector(
      TensorList tensors);
  static SmallVector<int64_t, N> ConvertIntArrayRefToSmallVector(
      IntArrayRef intArray);
  static SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
      const SmallVector<Tensor, N>& inputTensor);
  static SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
      const SmallVector<Tensor, N>& inputTensor,
      const SmallVector<uint, N>& masks);
  static SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
      const SmallVector<Scalar, N>& inputScalar,
      ScalarType scalar_type);
  static SmallVector<NPUTensorDesc, N> create_npu_output_tensor_desc(
      const SmallVector<Tensor, N>& outputTensor);
  static aclopAttr* CreateNpuAttrDesc(const SmallVector<NPUAttrDesc, N>& attrs);
  static NPUStatus CreateAclTensorDescInfo(
      SmallVector<NPUTensorDesc, N>& input,
      SmallVector<NPUTensorDesc, N>& output,
      ACL_PARAMS& params,
      string opName,
      const SmallVector<NPUAttrDesc, N>& attrs);
  static void execute_npu_operate(
      string opName,
      SmallVector<NPUTensorDesc, N>& inputs,
      SmallVector<NPUTensorDesc, N>& outputs,
      const SmallVector<NPUAttrDesc, N>& attrs);

  static SmallVector<int64_t, N> get_dimlist_for_tensor(const Tensor& self);
  static int64_t completePad(
      int64_t s_size,
      int64_t p_size,
      int64_t k_size,
      int64_t stride);
};

} // namespace npu
} // namespace native
} // namespace at

#endif
