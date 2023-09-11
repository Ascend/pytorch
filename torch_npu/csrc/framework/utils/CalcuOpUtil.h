#ifndef __PLUGIN_NATIVE_NPU_UTILS_CALCU_OP_UTIL__
#define __PLUGIN_NATIVE_NPU_UTILS_CALCU_OP_UTIL__

#include <cstdint>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl.h"


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

#define ACL_REQUIRE_OK_OP(expr, opstr)                                         \
  do {                                                                         \
    if (ASCEND_UNLIKELY((expr) != 0)) {                                        \
      printf("%s\n", opstr);                                                   \
      TORCH_CHECK((expr) == 0, __func__, ":", __FILE__, ":", __LINE__,         \
                  " NPU error,NPU error code is:", expr, "\n",                 \
                  c10_npu::acl::AclGetErrMsg());                               \
    }                                                                          \
  } while (0)

using StorageAndOffsetMemSizePair =
    std::pair<const c10::StorageImpl *, int64_t>;

namespace at_npu {
namespace native {

class CalcuOpUtil {
public:
  static aclDataType ConvertToAclDataType(const at::ScalarType &data_type);
  static aclDataType ConvertToAclDataType(const at::ScalarType &data_type,
                                              const string &realDataType);
  static c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor);
  static at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar,
                                       at::ScalarType scalar_data_type);
  static at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor);
  static NPUStatus AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t> &dst,
                                    size_t dst_size,
                                    const std::pair<at::Tensor, int64_t> &src,
                                    size_t src_size, aclrtMemcpyKind kind);

  // Add some public interfaces for aclrtmemcpy process,
  // to launch graph in graph mode automatically.
  static aclError
  AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst,
                            size_t dstMax,
                            const StorageAndOffsetMemSizePair &src,
                            size_t count, aclrtMemcpyKind kind);
  static aclError
  AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst,
                            size_t dstMax, const void *src, size_t count,
                            aclrtMemcpyKind kind);
  static aclError
  AclrtMemcpyWithModeSwitch(void *dst, size_t dstMax,
                            const StorageAndOffsetMemSizePair &src,
                            size_t count, aclrtMemcpyKind kind);
  static aclError LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor &dst,
                                                    size_t dstMax,
                                                    const at::Tensor &src,
                                                    size_t count,
                                                    aclrtMemcpyKind kind);
  static aclError LaunchAsyncCopyTaskWithModeSwitch(const c10::StorageImpl &dst,
                                                    size_t dstMax, void *src,
                                                    size_t count,
                                                    aclrtMemcpyKind kind);

  static void CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs,
                                  c10::ArrayRef<at::Tensor> outputs);
  static int64_t MakeWrapDim(int64_t dim, int64_t dim_post_expr);
  static bool IsTransposeLastTwoDims(const at::Tensor &tensor);
  static bool IsMmTranspose(const at::Tensor &tensor);
  static bool IsScalarWrappedToTensor(const at::Tensor &tensor);
  static bool IsNdToNzOnTheFly(const at::Tensor &self, const at::Tensor &mat2);
  static bool IsTransposeInnerAxis(const at::Tensor &self);
  static bool IsTransposeBothInnerAxis(const at::Tensor &self, const at::Tensor &mat2);
  static bool IsScalarOne(const c10::Scalar &scalar);
  static float GetScalarFloatValue(const c10::Scalar &scalar);
  static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
  static string GetReductionStr(int64_t reduction);
  static c10::SmallVector<at::Tensor, N>
  ConvertTensorListToSmallVector(at::TensorList tensors);
  static c10::SmallVector<int64_t, N>
  ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray);
  static c10::SmallVector<int64_t, N>
  GetDimlistForTensor(const at::Tensor &self);
  static int64_t CompletePad(int64_t s_size, int64_t p_size, int64_t k_size,
                             int64_t stride);
  static c10::SmallVector<int64_t, 3>
  ComputeOutputSize(c10::IntArrayRef input_size,
                      at::OptionalIntArrayRef output_size,
                      c10::optional<c10::ArrayRef<double>> scale_factors);
  static c10::optional<double>
  GetScaleValue(c10::optional<c10::ArrayRef<double>> scales, int idx);

  static int8_t GetCubeMathType(bool allowHf32);
};

} // namespace native
} // namespace at_npu

#endif
