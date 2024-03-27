#ifndef __PULGIN_NATIVE_UTILS_STORAGE_DESC_HELPER__
#define __PULGIN_NATIVE_UTILS_STORAGE_DESC_HELPER__

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>

#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/NPUDefinition.h"

namespace at_npu {
namespace native {

class StorageDescHelper {
public:
    // Get Part
    // sizes, strides in StorageDesc are same as those in MetaData
    static bool MetaDataAreMatch(const at::Tensor *tensor);
    // storage offset are match, the npu only support offset == 0
    static inline bool OffsetAreMatch(const at::Tensor *tensor) {return tensor->storage_offset() == 0;};

    // helper function of transdata op.
    static bool IsSameDesc(const torch_npu::NPUStorageDesc &a, const torch_npu::NPUStorageDesc &b);
    static bool IsSameDesc(const at::Tensor &a, const at::Tensor &b);

    // calculate storage size need by npu memory
    static int64_t GetMemorySize(const at::Tensor &dst);
    static int64_t GetMemorySize(const c10::IntArrayRef& size, aclFormat format, caffe2::TypeMeta dtype);
    // Calculate the valid memory size of the tensor, because of view operator and so on.
    static int64_t GetValidMemorySize(const at::Tensor &tensor);

    // Set Part
    // StorageDesc Init/Set
    static void SetDesc(at::Tensor &dst);
    static void SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides);
    static void SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, aclFormat format);
    static bool CheckDescInit(const c10::Storage &storage);

    // For Serialization to Get and Set NpuStorageDesc
    static void GetDescForSerialization(const at::Tensor &dst, std::unordered_map<std::string, bool> &desc_map);
    static void SetDescForSerialization(const at::Tensor &dst, std::unordered_map<std::string, bool> &desc_map);

    static void CopyDesc(at::Tensor &dst, const at::Tensor &src);
    static void CopyDesc(at::Tensor &dst, const c10::Storage &src);
    static void CopyDesc(const at::Tensor &dst, const torch_npu::NPUStorageDesc &src_desc);

    static void UpdateDesc(torch_npu::NPUStorageDesc &npuDesc, const c10::IntArrayRef &new_data_sizes,
                           const c10::IntArrayRef &new_shape_sizes);

    static FormatShape ComputeStrideFromShape(const FormatShape &shape);

    // need to remove later
    static void ReflushDescBySelf(const at::Tensor &src);

private:
    // Get Part
    static bool IsSameSize(const c10::SmallVector<int64_t, 5> &a, const c10::IntArrayRef &b);
    static int64_t GetMemorySize(const torch_npu::NPUStorageDesc &dst);
    // Set Part
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype);
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                             const c10::IntArrayRef& strides);
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                             const c10::IntArrayRef& strides, aclFormat format);
};

} // namespace native
} // namespace at_npu

#endif