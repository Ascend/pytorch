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

#ifndef __PLUGIN_NATIVE_NPU_UTILS_CALCU_OP_UTIL__
#define __PLUGIN_NATIVE_NPU_UTILS_CALCU_OP_UTIL__

#include <string>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include <stdint.h>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"


using std::string;
using std::vector;

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define ASCEND_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define ASCEND_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define ASCEND_LIKELY(expr) (expr)
#define ASCEND_UNLIKELY(expr) (expr)
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define ASCEND_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#elif defined(_MSC_VER)
#define ASCEND_ALWAYS_INLINE __forceinline
#else
#define ASCEND_ALWAYS_INLINE inline
#endif

#define ACL_REQUIRE_OK_OP(expr, opstr)     \
  do                                       \
  {                                        \
    if (ASCEND_UNLIKELY((expr) != 0))      \
    {                                      \
      printf("%s\n", opstr);               \
      TORCH_CHECK(                         \
          (expr) == 0,                     \
          __func__,                        \
          ":",                             \
          __FILE__,                        \
          ":",                             \
          __LINE__,                        \
          " NPU error,NPU error code is:", \
          expr, "\n",                      \
          c10_npu::acl::AclGetErrMsg());  \
    }                                      \
  } while (0)

using StorageAndOffsetMemSizePair = std::pair<const c10::StorageImpl*, int64_t>;

namespace at_npu
{
  namespace native
  {

    class NPUTensorDesc
    {
    public:
      enum TensorDescType
      {
        TENSOR = 0,
        SCALAR = 1,
        TENSOR_SCALAR = 2, // dim = 0 tensor
        NONE_TENSOR = 3,   // None at::Tensor
      };

    public:
      NPUTensorDesc() {}
      ~NPUTensorDesc() = default;

      explicit NPUTensorDesc(const at::Tensor &tensor)
          : tensor(tensor), tensorDescType(TensorDescType::TENSOR) {}

      explicit NPUTensorDesc(const c10::Scalar &scalar)
          : scalar(scalar), tensorDescType(TensorDescType::SCALAR) {}

      explicit NPUTensorDesc(const c10::Scalar &scalar, at::ScalarType scalarDataType)
          : scalar(scalar),
            scalarType(scalarDataType),
            tensorDescType(TensorDescType::SCALAR) {}

    public:
      at::Tensor tensor;
      c10::Scalar scalar;
      at::ScalarType scalarType = at::ScalarType::Undefined;
      TensorDescType tensorDescType;
      string tensorDescName;
      string realDataType;
    };

    class NPUAttrDesc
    {
    public:
      enum AttrDescType
      {
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
          : attrName(attrName), boolAttrValue(attrValue)
      {
        attrType = AttrDescType::BOOL_TYPE;
      }

      NPUAttrDesc(string attrName, int64_t attrValue)
          : attrName(attrName), intAttrValue(attrValue)
      {
        attrType = AttrDescType::INT_TYPE;
      }

      NPUAttrDesc(string attrName, float attrValue)
          : attrName(attrName), floatAttrValue(attrValue)
      {
        attrType = AttrDescType::FLOAT_TYPE;
      }

      NPUAttrDesc(string attrName, string attrValue)
          : attrName(attrName), stringAttrValue(attrValue)
      {
        attrType = AttrDescType::STRING_TYPE;
      }

      NPUAttrDesc(string attrName, c10::IntArrayRef attrValue) : attrName(attrName)
      {
        for (int i = 0; i < attrValue.size(); i++)
        {
          listIntAttrValue.emplace_back(attrValue[i]);
        }
        attrType = AttrDescType::LIST_INT_TYPE;
      }

      NPUAttrDesc(string attrName, at::ArrayRef<float> attrValue)
          : attrName(attrName)
      {
        for (int i = 0; i < attrValue.size(); i++)
        {
          listFloatAttrValue.emplace_back(attrValue[i]);
        }
        attrType = AttrDescType::LIST_FLOAT_TYPE;
      }

      NPUAttrDesc(string attrName, at::ArrayRef<c10::IntArrayRef> attrValue)
          : attrName(attrName)
      {
        c10::SmallVector<int64_t, N> listInt;

        for (int i = 0; i < attrValue.size(); i++)
        {
          for (int j = 0; j < attrValue[i].size(); j++)
          {
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
      c10::SmallVector<int64_t, N> listIntAttrValue;
      c10::SmallVector<float, N> listFloatAttrValue;
      c10::SmallVector<int64_t *, N>
          listListIntAttrValue; // Pointer to values of each listInt.
      c10::SmallVector<int, N>
          listListIntAttrListIntNum; // Pointer to number of each listInt.
      c10::SmallVector<c10::SmallVector<int64_t, N>, N>
          listListIntAttrListIntVal; // Value of each listInt.
    };

    class CalcuOpUtil
    {
    public:
      static aclDataType convert_to_acl_data_type(const at::ScalarType &data_type);
      static aclDataType convert_to_acl_data_type(
          const at::ScalarType &data_type,
          const string &realDataType);
      static at::ScalarType convert_to_at_data_type(const aclDataType acl_type);
      static c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor);
      static at::Tensor CopyScalarToDevice(
          const c10::Scalar &cpu_scalar,
          at::ScalarType scalar_data_type);
      static at::Tensor copy_tensor_host_to_device(const at::Tensor &cpu_tensor);
      static NPUStatus AclrtMemcpyAsync(
          const std::pair<at::Tensor, int64_t>& dst,
          size_t dst_size,
          const std::pair<at::Tensor, int64_t>& src,
          size_t src_size,
          aclrtMemcpyKind kind);

      // Add some public interfaces for aclrtmemcpy process,
      // to launch graph in graph mode automatically.
      static aclError AclrtMemcpyWithModeSwitch(
          const StorageAndOffsetMemSizePair& dst,
          size_t dstMax,
          const StorageAndOffsetMemSizePair& src,
          size_t count,
          aclrtMemcpyKind kind);
      static aclError AclrtMemcpyWithModeSwitch(
          const StorageAndOffsetMemSizePair& dst,
          size_t dstMax,
          const void* src,
          size_t count,
          aclrtMemcpyKind kind);
      static aclError AclrtMemcpyWithModeSwitch(
          void* dst,
          size_t dstMax,
          const StorageAndOffsetMemSizePair& src,
          size_t count,
          aclrtMemcpyKind kind);
      static aclError LaunchAsyncCopyTaskWithModeSwitch(
          const at::Tensor& dst,
          size_t dstMax,
          const at::Tensor& src,
          size_t count,
          aclrtMemcpyKind kind);
      static aclError LaunchAsyncCopyTaskWithModeSwitch(
          const c10::StorageImpl& dst,
          size_t dstMax,
          void* src,
          size_t count,
          aclrtMemcpyKind kind);

      static void check_memory_over_laps(
          c10::ArrayRef<at::Tensor> inputs,
          c10::ArrayRef<at::Tensor> outputs);
      static int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr);
      static bool is_transpose_last_two_dims(const at::Tensor &tensor);
      static bool is_scalar_wrapped_to_tensor(const at::Tensor &tensor);
      static bool is_scalar_one(const c10::Scalar &scalar);
      static float get_scalar_float_value(const c10::Scalar &scalar);
      static int64_t get_tensor_npu_format(const at::Tensor &tensor);
      static int64_t judge_and_get_format_from_input(bool is_cast_weight,
                                                     const at::Tensor &input,
                                                     int64_t target_format);
      static string get_reduction_str(int64_t reduction);
      static at::ScalarType GetNPUTensorDescScalarType(
          const NPUTensorDesc &npuTensorDesc);
      static c10::SmallVector<at::Tensor, N> ConvertTensorListToSmallVector(
          at::TensorList tensors);
      static c10::SmallVector<int64_t, N> ConvertIntArrayRefToSmallVector(
          c10::IntArrayRef intArray);
      static c10::SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
          const c10::SmallVector<at::Tensor, N> &inputTensor);
      static c10::SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
          const c10::SmallVector<at::Tensor, N> &inputTensor,
          const c10::SmallVector<uint, N> &masks);
      static c10::SmallVector<NPUTensorDesc, N> create_npu_input_tensor_desc(
          const c10::SmallVector<c10::Scalar, N> &inputScalar,
          at::ScalarType scalar_type);
      static c10::SmallVector<NPUTensorDesc, N> create_npu_output_tensor_desc(
          const c10::SmallVector<at::Tensor, N> &outputTensor);
      static aclopAttr* CreateNpuAttrDesc(const c10::SmallVector<NPUAttrDesc, N> &attrs);
      static NPUStatus CreateAclTensorDescInfo(
          c10::SmallVector<NPUTensorDesc, N> &input,
          c10::SmallVector<NPUTensorDesc, N> &output,
          ACL_PARAMS &params,
          string opName,
          const c10::SmallVector<NPUAttrDesc, N> &attrs);
      static void execute_npu_operate(
          string opName,
          c10::SmallVector<NPUTensorDesc, N> &inputs,
          c10::SmallVector<NPUTensorDesc, N> &outputs,
          const c10::SmallVector<NPUAttrDesc, N> &attrs);

      static c10::SmallVector<int64_t, N> get_dimlist_for_tensor(const at::Tensor &self);
      static int64_t completePad(int64_t s_size, int64_t p_size, int64_t k_size, int64_t stride);
      static c10::SmallVector<int64_t, 3> compute_output_size(
          c10::IntArrayRef input_size,
          c10::optional<c10::IntArrayRef> output_size,
          c10::optional<c10::ArrayRef<double>> scale_factors);
      static c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx);
    };

  } // namespace native
} // namespace at_npu

#endif
