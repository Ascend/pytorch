#ifndef __PULGIN_NATIVE_UTILS_FORMAT_HELPER__
#define __PULGIN_NATIVE_UTILS_FORMAT_HELPER__

#include <ATen/ATen.h>
#include <unordered_map>

#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu
{
  namespace native
  {
    using baseFormatConverter = std::function<FormatShape(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims)>;
    // helper function of storage format
    class FormatHelper
    {
    public:
      // helper function of copy, because of padding will change the physical size.
      static bool IsPadded(const at::Tensor *tensor);
      static char *GetFormatName(const at::Tensor &tensor);
      static char *GetFormatName(aclFormat format);
      static aclFormat GetBaseFormat(const at::Tensor &tensor);
      static aclFormat GetBaseFormat(aclFormat format);
      static aclFormat GetFormat(const at::Tensor &tensor);

      static bool IsBaseFormatType(aclFormat format);
      static bool IsBaseFormatType(const at::Tensor &tensor);

      // Default assumption: the original format are ND, NCHW or NDHWC.
      // So, if original size are 4D, it maybe NCHW or ND and so on.
      // The format can be split into two parts:
      // 1. The storage size can be infered between NC1HWC0, NHWC, NC1HWC0_C04, NCHW.
      // 2. The storage size can be infered between NDC1HWC0 and NDHWC/NCDHW.
      // The storage size can not be infered between different groups.
      template <typename sizeType>
      static FormatShape GetStorageSizes(aclFormat format, sizeType ori_size);
      // GetStorageSizes used to calculate the storage sizes of op at npu device at different format.
      static FormatShape GetStorageSizes(const torch_npu::NPUStorageDesc &desc);
      static at::Tensor& unsafe_format_cast(at::Tensor& self, int64_t self_format, int64_t result_format);

      static bool IsOpInputBaseFormat(const at::Tensor &tensor);
      static bool IsOpInputBaseFormat(const c10::optional<at::Tensor> &tensor);
      static bool IsOpInputBaseFormat(const c10::List<c10::optional<at::Tensor>> &tensors);
      static bool IsOpInputBaseFormat(const at::TensorList &tensors);
      static bool IsOpInputBaseFormat(const at::ITensorListRef &tensors);

    private:
      static bool IsPadded(aclFormat format);

    private:
      using shapeInfer = std::function<FormatShape(c10::IntArrayRef dims)>;
      typedef struct FormatInfo_
      {
        aclFormat format = ACL_FORMAT_ND;
        aclFormat baseFormat = ACL_FORMAT_ND;
        shapeInfer func = nullptr;
        char formatName[30] = {0};
        bool isPadded = false;
      } FormatInfo;
      static std::unordered_map<aclFormat, FormatInfo> info;
    }; // class FormatHelper

    // template impl
    template <typename sizeType>
    FormatShape FormatHelper::GetStorageSizes(aclFormat format, sizeType ori_size)
    {
      auto itr = info.find(format);
      if (itr != info.end())
      {
        if (itr->second.func)
        {
          return itr->second.func(ori_size);
        }
      }
      AT_ERROR("unsupport InferShape with format ", GetFormatName(format), "with shape", ori_size);
      return {};
    }

  } // native
} // at
#endif